"""
Training Loop for RL Agent

Implements PPO training with:
- Episode collection with multi-step reformulation
- Experience replay buffer
- PPO policy updates
- Evaluation on validation set
- Checkpointing and logging
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
from collections import deque, defaultdict
from tqdm import tqdm
import time

from ..rl_agent import QueryReformulatorAgent, RLTrainer
from ..pipeline import AdaptiveIRPipeline
from ..evaluation import IRMetricsAggregator
from ..utils import setup_logging, save_checkpoint, load_checkpoint, EarlyStopping


class ReplayBuffer:
    """Experience replay buffer for PPO."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        query_emb: torch.Tensor,
        current_query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        candidate_features: torch.Tensor,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool
    ):
        """Add experience tuple."""
        experience = {
            'query_emb': query_emb.cpu(),
            'current_query_emb': current_query_emb.cpu(),
            'candidate_embs': candidate_embs.cpu(),
            'candidate_features': candidate_features.cpu(),
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()


class RLTrainingLoop:
    """
    Main training loop for RL query reformulation.
    """
    
    def __init__(
        self,
        config: Dict,
        pipeline: AdaptiveIRPipeline,
        train_dataset,
        val_dataset,
        test_dataset=None
    ):
        """
        Args:
            config: Configuration dictionary
            pipeline: AdaptiveIRPipeline instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset
        """
        self.config = config
        self.pipeline = pipeline
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        
        # Training parameters
        self.device = config.get('system', {}).get('device', 'cuda')
        self.num_epochs = config.get('training', {}).get('num_epochs', 100)
        self.batch_size = config.get('training', {}).get('batch_size', 32)
        self.episodes_per_update = config.get('training', {}).get('episodes_per_update', 128)
        self.ppo_epochs = config.get('training', {}).get('ppo_epochs', 4)
        self.max_steps = config.get('rl_agent', {}).get('max_steps_per_episode', 5)
        
        # Reward shaping
        self.reward_weights = config.get('training', {}).get('reward_weights', {
            'recall': 0.7,
            'mrr': 0.3
        })
        
        # RL Trainer
        self.rl_trainer = RLTrainer(
            agent=pipeline.rl_agent,
            config=config['rl_agent']
        )
        
        # Replay buffer
        buffer_size = config.get('training', {}).get('buffer_size', 10000)
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('training', {}).get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_freq = config.get('training', {}).get('save_freq', 5)
        
        # Early stopping
        patience = config.get('training', {}).get('early_stopping_patience', 10)
        self.early_stopping = EarlyStopping(patience=patience, mode='max')
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.metrics_history = defaultdict(list)
        
        # Baseline metrics (without RL)
        self.baseline_metrics = None
        
        self.logger.info("Initialized RL Training Loop")
    
    def compute_reward(
        self,
        metrics_before: Dict[str, float],
        metrics_after: Dict[str, float]
    ) -> float:
        """
        Compute shaped reward based on metric improvements.
        
        Reward = w_recall * ΔRecall@100 + w_mrr * ΔMRR@10
        
        Args:
            metrics_before: Metrics before reformulation
            metrics_after: Metrics after reformulation
            
        Returns:
            Reward scalar
        """
        delta_recall = metrics_after.get('recall@100', 0.0) - metrics_before.get('recall@100', 0.0)
        delta_mrr = metrics_after.get('mrr', 0.0) - metrics_before.get('mrr', 0.0)
        
        reward = (
            self.reward_weights['recall'] * delta_recall +
            self.reward_weights['mrr'] * delta_mrr
        )
        
        return reward
    
    def collect_episode(
        self,
        query: str,
        qrels: Dict[str, int]
    ) -> Tuple[List[Dict], float]:
        """
        Collect one episode (full reformulation trajectory).
        
        Args:
            query: Original query string
            qrels: Ground-truth relevance judgments {doc_id: relevance}
            
        Returns:
            (trajectory, total_reward)
        """
        trajectory = []
        
        # Step 0: Evaluate original query
        result_before = self.pipeline.search(query, top_k=100)
        doc_ids_before = [doc_id for doc_id, _ in result_before['results']]
        
        evaluator = IRMetricsAggregator()
        metrics_before = evaluator.compute_single_query(
            qrels,
            doc_ids_before,
            query_id=None
        )
        
        # Stage 0: Mine candidates
        candidates = self.pipeline.mine_candidates(query)
        
        if not candidates:
            # No candidates: return empty trajectory
            return trajectory, 0.0
        
        # Prepare inputs
        query_emb = self.pipeline._embed_text(query)
        candidate_terms = list(candidates.keys())
        candidate_embs = torch.stack([self.pipeline._embed_text(term) for term in candidate_terms])
        feature_matrix = self.pipeline.candidate_miner.get_candidate_features(candidates)
        candidate_features = torch.tensor(feature_matrix, dtype=torch.float32)
        
        # Move to device
        query_emb = query_emb.to(self.device)
        candidate_embs = candidate_embs.to(self.device)
        candidate_features = candidate_features.to(self.device)
        
        # Episode: select terms iteratively
        current_query = query
        selected_terms = []
        total_reward = 0.0
        
        for step in range(self.max_steps):
            current_emb = self.pipeline._embed_text(current_query).unsqueeze(0).to(self.device)
            
            # Select action
            action, log_prob, value = self.pipeline.rl_agent.select_action(
                query_emb.unsqueeze(0),
                current_emb,
                candidate_embs.unsqueeze(0),
                candidate_features.unsqueeze(0),
                deterministic=False
            )
            
            action_idx = action.item()
            log_prob_val = log_prob.item()
            value_val = value.item()
            
            # Check if STOP action
            done = (action_idx >= len(candidate_terms))
            
            if not done:
                # Add selected term
                selected_term = candidate_terms[action_idx]
                selected_terms.append(selected_term)
                current_query = current_query + " " + selected_term
            
            # Evaluate reformulated query
            result_after = self.pipeline.search(current_query, top_k=100)
            doc_ids_after = [doc_id for doc_id, _ in result_after['results']]
            
            metrics_after = evaluator.compute_single_query(
                qrels,
                doc_ids_after,
                query_id=None
            )
            
            # Compute reward
            reward = self.compute_reward(metrics_before, metrics_after)
            total_reward += reward
            
            # Store experience
            experience = {
                'query_emb': query_emb,
                'current_query_emb': current_emb.squeeze(0),
                'candidate_embs': candidate_embs,
                'candidate_features': candidate_features,
                'action': action_idx,
                'log_prob': log_prob_val,
                'value': value_val,
                'reward': reward,
                'done': done
            }
            trajectory.append(experience)
            
            # Update metrics for next step
            metrics_before = metrics_after
            
            if done:
                break
        
        return trajectory, total_reward
    
    def update_policy(self):
        """
        Update RL agent using PPO.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        query_embs = torch.stack([exp['query_emb'] for exp in batch]).to(self.device)
        current_query_embs = torch.stack([exp['current_query_emb'] for exp in batch]).to(self.device)
        candidate_embs = torch.stack([exp['candidate_embs'] for exp in batch]).to(self.device)
        candidate_features = torch.stack([exp['candidate_features'] for exp in batch]).to(self.device)
        
        actions = torch.tensor([exp['action'] for exp in batch], dtype=torch.long).to(self.device)
        old_log_probs = torch.tensor([exp['log_prob'] for exp in batch], dtype=torch.float32).to(self.device)
        rewards = torch.tensor([exp['reward'] for exp in batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([exp['done'] for exp in batch], dtype=torch.bool).to(self.device)
        
        # Get values
        values = torch.tensor([exp['value'] for exp in batch], dtype=torch.float32).to(self.device)

        # Prepare rollout buffer for PPO update
        rollout_buffer = {
            'query_emb': query_embs,
            'current_query_emb': current_query_embs,
            'candidate_embs': candidate_embs,
            'candidate_features': candidate_features,
            'actions': actions,
            'old_log_probs': old_log_probs,
            'rewards': rewards.unsqueeze(1),  # [batch, 1] for GAE computation
            'dones': dones.unsqueeze(1),  # [batch, 1] for GAE computation
            'values': values.unsqueeze(1)  # [batch, 1] for GAE computation
        }

        # PPO update
        loss = self.rl_trainer.update(rollout_buffer)

        return loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Epoch number
            
        Returns:
            Training metrics
        """
        self.pipeline.enable_training_mode()
        
        epoch_rewards = []
        epoch_losses = []
        episode_count = 0
        
        # Sample queries
        train_queries = self.train_dataset.load_queries()
        train_qrels = self.train_dataset.load_qrels()
        
        query_ids = list(train_queries.keys())
        np.random.shuffle(query_ids)
        
        pbar = tqdm(query_ids, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for query_id in pbar:
            query = train_queries[query_id]
            qrels = train_qrels.get(query_id, {})
            
            if not qrels:
                continue
            
            # Collect episode
            trajectory, total_reward = self.collect_episode(query, qrels)
            
            if trajectory:
                # Add to replay buffer
                for experience in trajectory:
                    self.replay_buffer.push(**experience)
                
                epoch_rewards.append(total_reward)
                episode_count += 1
                
                # Update policy every N episodes
                if episode_count % self.episodes_per_update == 0:
                    for _ in range(self.ppo_epochs):
                        loss = self.update_policy()
                        if loss is not None:
                            epoch_losses.append(loss)
                
                pbar.set_postfix({
                    'reward': f"{np.mean(epoch_rewards[-100:]):.4f}",
                    'episodes': episode_count
                })

        # Aggregate metrics
        metrics = {
            'avg_reward': np.mean(epoch_rewards) if epoch_rewards else 0.0,
            'num_episodes': episode_count
        }

        # Aggregate loss metrics from dicts
        if epoch_losses:
            # epoch_losses is a list of dicts, need to aggregate each metric
            loss_keys = epoch_losses[0].keys()
            for key in loss_keys:
                values = [loss_dict[key] for loss_dict in epoch_losses]
                metrics[f'avg_{key}'] = np.mean(values)

        return metrics
    
    def evaluate(self, dataset, split: str = 'val') -> Dict[str, float]:
        """
        Evaluate on validation/test set.
        
        Args:
            dataset: Dataset to evaluate
            split: 'val' or 'test'
            
        Returns:
            Evaluation metrics
        """
        self.pipeline.enable_eval_mode()
        
        queries = dataset.load_queries()
        qrels = dataset.load_qrels()
        
        evaluator = IRMetricsAggregator()
        
        for query_id, query in tqdm(queries.items(), desc=f"Evaluating {split}"):
            qrel = qrels.get(query_id, {})
            
            if not qrel:
                continue
            
            # Run pipeline
            result = self.pipeline.search(query, top_k=100)
            doc_ids = [doc_id for doc_id, _ in result['results']]

            # Add to evaluator
            # qrel is dict mapping doc_id -> relevance
            relevant_set = set(qrel.keys())
            evaluator.add_query_result(
                query_id=query_id,
                retrieved=doc_ids,
                relevant=relevant_set,
                relevant_grades=qrel
            )

        # Compute aggregate metrics
        metrics = evaluator.compute_aggregate()

        return metrics
    
    def train(self):
        """
        Main training loop.
        """
        self.logger.info("Starting training...")
        
        # Baseline evaluation (without RL)
        self.logger.info("Computing baseline metrics (no RL)...")
        # TODO: Implement baseline evaluation with original queries
        
        best_metric = 0.0
        
        for epoch in range(self.num_epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Build loss string
            loss_str = ""
            if 'avg_policy_loss' in train_metrics:
                loss_str = (f"Policy Loss: {train_metrics['avg_policy_loss']:.4f} | "
                           f"Value Loss: {train_metrics['avg_value_loss']:.4f}")

            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs} | "
                           f"Reward: {train_metrics['avg_reward']:.4f} | "
                           f"{loss_str}")
            
            # Validation
            if (epoch + 1) % self.save_freq == 0:
                val_metrics = self.evaluate(self.val_dataset, 'val')
                
                self.logger.info(f"Validation | "
                               f"Recall@100: {val_metrics['recall@100']:.4f} | "
                               f"MRR: {val_metrics['mrr']:.4f}")

                # Save checkpoint
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(
                    self.pipeline.rl_agent,
                    self.rl_trainer.optimizer,
                    epoch,
                    val_metrics,
                    checkpoint_path
                )

                # Early stopping
                current_metric = val_metrics['mrr']
                
                if self.early_stopping(current_metric):
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
                
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
                    save_checkpoint(
                        self.pipeline.rl_agent,
                        self.rl_trainer.optimizer,
                        epoch,
                        val_metrics,
                        best_checkpoint_path
                    )
                    self.logger.info(f"Saved best model with MRR: {best_metric:.4f}")
            
            # Store metrics
            self.metrics_history['train_reward'].append(train_metrics['avg_reward'])
            # Store policy loss if available, otherwise 0
            loss_value = train_metrics.get('avg_policy_loss', 0.0)
            self.metrics_history['train_loss'].append(loss_value)
        
        # Final test evaluation
        if self.test_dataset:
            self.logger.info("Running final test evaluation...")
            
            # Load best model
            best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
            load_checkpoint(self.pipeline.rl_agent, best_checkpoint_path)
            
            test_metrics = self.evaluate(self.test_dataset, 'test')
            
            self.logger.info(f"Test Results | "
                           f"Recall@100: {test_metrics['recall@100']:.4f} | "
                           f"MRR: {test_metrics['mrr']:.4f}")
            
            # Save test metrics
            test_results_path = self.checkpoint_dir / "test_results.json"
            with open(test_results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
        
        self.logger.info("Training completed!")


if __name__ == "__main__":
    print("RL Training Loop module loaded successfully")

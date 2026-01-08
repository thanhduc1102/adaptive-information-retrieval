"""
RL Query Reformulation Agent

Actor-Critic architecture with Transformer encoder for query reformulation.
Converted from legacy Theano implementation to modern PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class QueryReformulatorAgent(nn.Module):
    """
    Actor-Critic RL Agent for Query Reformulation.
    
    Architecture:
        Input: [query_emb, current_query_emb, candidate_features]
        ↓
        Transformer Encoder
        ↓
        Attention(current_query, candidates)
        ↓
        Actor Head (policy) + Critic Head (value)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Dimensions
        self.embedding_dim = config.get('embedding_dim', 512)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_heads = config.get('num_attention_heads', 4)
        self.num_layers = config.get('num_encoder_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        
        # Query encoders
        self.query_encoder = nn.Linear(self.embedding_dim, self.hidden_dim)
        
        # Candidate feature encoder
        self.candidate_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim + 6, self.hidden_dim),  # +6 for features
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)  # Logit for each candidate
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # STOP action head
        self.stop_head = nn.Linear(self.hidden_dim, 1)
        
    def forward(
        self,
        query_emb: torch.Tensor,
        current_query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query_emb: Original query embedding [batch, emb_dim]
            current_query_emb: Current query embedding [batch, emb_dim]
            candidate_embs: Candidate embeddings [batch, num_cands, emb_dim]
            candidate_features: Candidate features [batch, num_cands, num_features]
            candidate_mask: Mask for padding [batch, num_cands]
            
        Returns:
            action_logits: Logits for each action [batch, num_cands + 1] (+1 for STOP)
            value: State value estimate [batch, 1]
            attention_weights: Attention over candidates [batch, num_cands]
        """
        batch_size, num_cands, _ = candidate_embs.shape
        
        # Encode queries
        q0_enc = self.query_encoder(query_emb)  # [batch, hidden]
        qt_enc = self.query_encoder(current_query_emb)  # [batch, hidden]
        
        # Encode candidates with features
        cand_input = torch.cat([candidate_embs, candidate_features], dim=-1)
        cand_enc = self.candidate_encoder(cand_input)  # [batch, num_cands, hidden]
        
        # Prepare input for transformer: [q0, qt, cand1, cand2, ...]
        seq_input = torch.cat([
            q0_enc.unsqueeze(1),  # [batch, 1, hidden]
            qt_enc.unsqueeze(1),  # [batch, 1, hidden]
            cand_enc  # [batch, num_cands, hidden]
        ], dim=1)  # [batch, 2 + num_cands, hidden]
        
        # Create attention mask if needed
        if candidate_mask is not None:
            # Extend mask for q0 and qt (never masked)
            full_mask = torch.cat([
                torch.ones(batch_size, 2, device=candidate_mask.device),
                candidate_mask
            ], dim=1).bool()
        else:
            full_mask = None
        
        # Transformer encoding
        encoded = self.transformer(seq_input, src_key_padding_mask=~full_mask if full_mask is not None else None)
        
        # Extract components
        qt_encoded = encoded[:, 1, :]  # Current query representation
        cand_encoded = encoded[:, 2:, :]  # Candidate representations
        
        # Attention: query attends to candidates
        attended, attn_weights = self.attention(
            qt_encoded.unsqueeze(1),  # Query: [batch, 1, hidden]
            cand_encoded,  # Key: [batch, num_cands, hidden]
            cand_encoded,  # Value: [batch, num_cands, hidden]
            key_padding_mask=~candidate_mask if candidate_mask is not None else None
        )
        attended = attended.squeeze(1)  # [batch, hidden]
        
        # Actor: compute action logits
        cand_logits = self.actor(cand_encoded).squeeze(-1)  # [batch, num_cands]
        stop_logit = self.stop_head(attended)  # [batch, 1]
        action_logits = torch.cat([cand_logits, stop_logit], dim=-1)  # [batch, num_cands + 1]
        
        # Apply mask to logits
        if candidate_mask is not None:
            # Mask out invalid candidates
            mask_value = -1e9
            action_logits[:, :-1] = action_logits[:, :-1].masked_fill(~candidate_mask, mask_value)
        
        # Critic: compute value
        value = self.critic(attended)  # [batch, 1]
        
        return action_logits, value, attn_weights.squeeze(1)
    
    def select_action(
        self,
        query_emb: torch.Tensor,
        current_query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy.
        
        Args:
            [same as forward]
            deterministic: If True, select argmax. If False, sample from distribution.
            
        Returns:
            action: Selected action indices [batch]
            log_prob: Log probability of selected action [batch]
            value: State value [batch, 1]
        """
        action_logits, value, attn_weights = self.forward(
            query_emb, current_query_emb, candidate_embs,
            candidate_features, candidate_mask
        )
        
        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        # Select action
        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        # Get log probability
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def evaluate_actions(
        self,
        query_emb: torch.Tensor,
        current_query_emb: torch.Tensor,
        candidate_embs: torch.Tensor,
        candidate_features: torch.Tensor,
        actions: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_probs: Log probabilities of actions [batch]
            values: State values [batch, 1]
            entropy: Entropy of action distribution [batch]
        """
        action_logits, value, _ = self.forward(
            query_emb, current_query_emb, candidate_embs,
            candidate_features, candidate_mask
        )
        
        # Create distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        # Evaluate actions
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, value, entropy


class RLTrainer:
    """
    PPO Trainer for RL Agent.
    """
    
    def __init__(self, agent: QueryReformulatorAgent, config: Dict):
        self.agent = agent
        self.config = config
        
        # Training hyperparameters
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.num_epochs = config.get('num_ppo_epochs', 4)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=self.lr)
        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards [batch, seq_len]
            values: Value estimates [batch, seq_len]
            dones: Done flags [batch, seq_len]
            
        Returns:
            advantages: Advantage estimates [batch, seq_len]
            returns: Target returns [batch, seq_len]
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(rewards.shape[1])):
            if t == rewards.shape[1] - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]
            
            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t]) - values[:, t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t]) * gae
            
            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]
        
        return advantages, returns
    
    def update(self, rollout_buffer: Dict) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            rollout_buffer: Dictionary containing:
                - observations: State observations
                - actions: Actions taken
                - old_log_probs: Log probs from old policy
                - rewards: Rewards received
                - dones: Episode done flags
                - values: Old value estimates
                
        Returns:
            Dictionary of training metrics
        """
        # Compute advantages
        advantages, returns = self.compute_gae(
            rollout_buffer['rewards'],
            rollout_buffer['values'],
            rollout_buffer['dones']
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(self.num_epochs):
            # Evaluate actions with current policy
            log_probs, values, entropy = self.agent.evaluate_actions(
                rollout_buffer['query_emb'],
                rollout_buffer['current_query_emb'],
                rollout_buffer['candidate_embs'],
                rollout_buffer['candidate_features'],
                rollout_buffer['actions'],
                rollout_buffer.get('candidate_mask')
            )
            
            # Ratio for PPO
            ratio = torch.exp(log_probs - rollout_buffer['old_log_probs'])
            
            # Clipped surrogate objective
            surr1 = ratio * advantages.flatten()
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.flatten()
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns.flatten())
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Accumulate metrics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
        
        # Return averaged metrics
        return {
            'policy_loss': total_policy_loss / self.num_epochs,
            'value_loss': total_value_loss / self.num_epochs,
            'entropy': total_entropy / self.num_epochs
        }


if __name__ == "__main__":
    # Test agent
    config = {
        'embedding_dim': 512,
        'hidden_dim': 256,
        'num_attention_heads': 4,
        'num_encoder_layers': 2,
        'dropout': 0.1
    }
    
    agent = QueryReformulatorAgent(config)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    num_cands = 50
    
    query_emb = torch.randn(batch_size, 512)
    current_query_emb = torch.randn(batch_size, 512)
    candidate_embs = torch.randn(batch_size, num_cands, 512)
    candidate_features = torch.randn(batch_size, num_cands, 6)
    
    action, log_prob, value = agent.select_action(
        query_emb, current_query_emb, candidate_embs, candidate_features
    )
    
    print(f"Action shape: {action.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Selected actions: {action}")

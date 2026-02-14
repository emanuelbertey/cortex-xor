/*!
# MinGRU Implementation

This module implements the MinGRU as described in the paper:
"MinGRU: Designing Simpler and More Efficient RNNs" (2024).

MinGRU is a simplified version of GRU that uses associative scanning
for parallel computation while maintaining the same performance.
*/

use candle_core::{Tensor, Result};
use candle_nn::{Module, VarBuilder, Linear, linear};
use serde::{Deserialize, Serialize};

/// Configuration for MinGRU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinGruConfig {
    /// Input dimension
    pub dim: usize,
    /// Expansion factor for inner dimension
    pub expansion_factor: f32,
    /// Whether to project output back to original dimension
    pub proj_out: bool,
}

impl MinGruConfig {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            expansion_factor: 1.0,
            proj_out: false,
        }
    }

    pub fn with_expansion_factor(mut self, expansion_factor: f32) -> Self {
        self.expansion_factor = expansion_factor;
        self.proj_out = expansion_factor != 1.0;
        self
    }

    pub fn with_proj_out(mut self, proj_out: bool) -> Self {
        self.proj_out = proj_out;
        self
    }

    /// Initialize a new MinGRU
    pub fn init(&self, vb: VarBuilder) -> Result<MinGru> {
        let dim_inner = (self.dim as f32 * self.expansion_factor) as usize;
        
        let to_hidden_and_gate = linear(self.dim, dim_inner * 2, vb.pp("to_hidden_and_gate"))?;
        
        let to_out = if self.proj_out {
            Some(linear(dim_inner, self.dim, vb.pp("to_out"))?)
        } else {
            None
        };

        Ok(MinGru {
            to_hidden_and_gate,
            to_out,
            dim_inner,
            dim: self.dim,
        })
    }
}

/// MinGRU cell implementation
#[derive(Debug)]
pub struct MinGru {
    /// Linear layer for hidden and gate projections
    pub to_hidden_and_gate: Linear,
    /// Optional output projection layer
    pub to_out: Option<Linear>,
    /// Inner dimension
    pub dim_inner: usize,
    /// Input dimension
    pub dim: usize,
}

impl MinGru {
    /// Softplus function for numerical stability
    fn softplus(&self, x: &Tensor) -> Result<Tensor> {
        // Softplus(x) = log(1 + exp(x))
        // For stability: max(0, x) + log(1 + exp(-|x|))
        let zeros = Tensor::zeros_like(x)?;
        let abs_x = x.abs()?;
        let neg_abs_x = abs_x.neg()?;
        let inner = neg_abs_x.exp()?.add(&Tensor::full(1.0f32, x.shape(), x.device())?)?.log()?;
        
        x.maximum(&zeros)?.add(&inner)
    }

    /// g function from appendix B.3
    fn g(&self, x: &Tensor) -> Result<Tensor> {
        // g(x) = torch.where(x >= 0, x + 0.5, x.sigmoid())
        let zeros = Tensor::zeros_like(x)?;
        let half = Tensor::full(0.5f32, x.shape(), x.device())?;
        let condition = x.ge(&zeros)?;
        
        let positive_case = x.add(&half)?;
        let negative_case = candle_nn::ops::sigmoid(x)?;
        
        Tensor::where_cond(&condition, &positive_case, &negative_case)
    }

    /// log_g function from appendix B.3
    fn log_g(&self, x: &Tensor) -> Result<Tensor> {
        // log_g(x) = torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))
        let zeros = Tensor::zeros_like(x)?;
        let condition = x.ge(&zeros)?;
        
        let positive_case = x.relu()?.add(&Tensor::full(0.5f32, x.shape(), x.device())?)?.log()?;
        let negative_case = self.softplus(&x.neg()?)?.neg()?;
        Tensor::where_cond(&condition, &positive_case, &negative_case)
    }

    /// Heinsen associative scan in log space (appendix B)
    fn heinsen_associative_scan_log(&self, log_coeffs: &Tensor, log_values: &Tensor) -> Result<Tensor> {
        let a_star = log_coeffs.cumsum(1)?;
        let log_ratio = log_values.sub(&a_star.broadcast_as(log_values.shape())?)?;
        let ratio_max = log_ratio.max_keepdim(1)?;
        let s = log_ratio.broadcast_sub(&ratio_max)?.exp()?.cumsum(1)?;
        let out_log = s.log()?.broadcast_add(&a_star)?.broadcast_add(&ratio_max)?;
        out_log.exp()
    }

    /// Forward pass for MinGRU
    pub fn forward(
        &self,
        x: &Tensor,
        prev_hidden: Option<&Tensor>,
        return_next_prev_hidden: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let seq_len = x.dims()[1];
        
        // hidden, gate = self.to_hidden_and_gate(x).chunk(2, dim = -1)
        let hidden_and_gate = self.to_hidden_and_gate.forward(x)?;
        let chunks = hidden_and_gate.chunk(2, 2)?;
        let hidden_raw = &chunks[0];
        let gate = &chunks[1];

        let gate = candle_nn::ops::sigmoid(gate)?;

        // Parallelizable linear recurrence:
        // h_t = a_t * h_{t-1} + b_t
        // a_t = 1 - gate_t
        // b_t = gate_t * hidden_t
        // (all terms are positive because g(x) > 0 and sigmoid > 0)
        let eps = Tensor::full(1e-6f32, gate.shape(), gate.device())?;
        let one = Tensor::full(1.0f32, gate.shape(), gate.device())?;
        let a = one.sub(&gate)?.maximum(&eps)?;
        let mut log_coeffs = a.log()?;
        let log_hidden = self.log_g(hidden_raw)?.clamp(-6.0, 6.0)?;
        let log_gate = gate.maximum(&eps)?.log()?;
        let mut log_values = log_gate.add(&log_hidden)?;

        // If we have a previous hidden state, prepend it as the first value with coeff=1.
        // This allows the scan to incorporate the initial condition.
        if let Some(h0) = prev_hidden {
            let h0_pos = h0.maximum(&Tensor::full(1e-6f32, h0.shape(), h0.device())?)?;
            let log_h0 = h0_pos.log()?; // [batch, 1, hidden]
            let log_a0 = Tensor::zeros_like(&log_h0)?; // log(1)
            log_coeffs = Tensor::cat(&[&log_a0, &log_coeffs], 1)?;
            log_values = Tensor::cat(&[&log_h0, &log_values], 1)?;
        }

        let mut out_tensor = self.heinsen_associative_scan_log(&log_coeffs, &log_values)?;

        // Drop the prepended initial state if present.
        if prev_hidden.is_some() {
            out_tensor = out_tensor.narrow(1, 1, seq_len)?;
        }

        let next_prev_hidden = out_tensor.narrow(1, out_tensor.dims()[1] - 1, 1)?;

        // Apply output projection if needed
        let final_out = if let Some(to_out) = &self.to_out {
            to_out.forward(&out_tensor)?
        } else {
            out_tensor
        };

        if return_next_prev_hidden {
            Ok((final_out, Some(next_prev_hidden)))
        } else {
            Ok((final_out, None))
        }
    }

    /// Forward pass without returning next hidden state
    pub fn forward_simple(&self, x: &Tensor, prev_hidden: Option<&Tensor>) -> Result<Tensor> {
        let (out, _) = self.forward(x, prev_hidden, false)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};

    #[test]
    fn test_min_gru_init() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let config = MinGruConfig::new(64).with_expansion_factor(2.0);
        let min_gru = config.init(vb)?;
        
        assert_eq!(min_gru.dim, 64);
        assert_eq!(min_gru.dim_inner, 128);
        assert!(min_gru.to_out.is_some());
        
        Ok(())
    }

    #[test]
    fn test_min_gru_forward() -> Result<()> {
        let device = Device::Cpu;
        let vb = VarBuilder::zeros(DType::F32, &device);
        
        let config = MinGruConfig::new(32);
        let min_gru = config.init(vb)?;
        
        let batch_size = 2;
        let seq_len = 10;
        let x = Tensor::randn(0.0, 1.0, (batch_size, seq_len, 32), &device)?;
        
        // Test forward pass
        let output = min_gru.forward_simple(&x, None)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, 32]);
        
        Ok(())
    }
}

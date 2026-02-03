/*
# RWKV-7 / mLSTM Hybrid

This module implements a hybrid RNN cell that uses the matrix state structure of mLSTM
but applies the Delta Rule update mechanism from RWKV-7.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, Module, VarBuilder, Linear, LayerNorm, ops, linear_no_bias, layer_norm};

/// State for the Hybrid Cell
#[derive(Clone, Debug)]
pub struct MLstmstate {
    /// Cell state - matrix of shape [`batch_size`, `num_heads`, `head_dim`, `head_dim`]
    pub cell: Tensor,
    /// Hidden state - vector of shape [`batch_size`, `hidden_size`]
    pub hidden: Tensor,
}

impl MLstmstate {
    /// Create a new state
    pub fn new(cell: Tensor, hidden: Tensor) -> Self {
        Self { cell, hidden }
    }

    pub fn detach(&self) -> Self {
        Self {
            cell: self.cell.detach(),
            hidden: self.hidden.detach(),
        }
    }
}

/// Configuration
#[derive(Debug, Clone)]
pub struct MLstmconfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub dropout: f32,
}

impl MLstmconfig {
    pub fn new(d_input: usize, d_hidden: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            num_layers,
            num_heads,
            dropout: 0.0,
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn init(&self, vb: VarBuilder) -> Result<MLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            let layer_vb = vb.pp(format!("layer_{}", i));
            layers.push(MLstmcell::new(input_size, self.d_hidden, self.num_heads, layer_vb)?);
        }

        Ok(MLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            dropout: self.dropout,
        })
    }
}

/// Main Layer
#[derive(Debug)]
pub struct MLstm {
    pub layers: Vec<MLstmcell>,
    pub dropout_layer: Dropout,
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub dropout: f32,
}

impl MLstm {
    pub fn forward(
        &self,
        input_seq: &Tensor,
        states: Option<Vec<MLstmstate>>,
    ) -> Result<(Tensor, Vec<MLstmstate>)> {
        let (batch_size, _seq_length, _) = input_seq.dims3()?;
        let device = input_seq.device();

        let mut hidden_states = match states {
            Some(s) => s,
            None => self.init_hidden(batch_size, device)?,
        };
        
        let mut layer_input = input_seq.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let old_state = &hidden_states[layer_idx];
            let (h_seq, new_state) = layer.forward_sequence(&layer_input, old_state)?;
            hidden_states[layer_idx] = new_state;

            layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                self.dropout_layer.forward(&h_seq, false)?
            } else {
                h_seq
            };
        }

        Ok((layer_input, hidden_states))
    }

    fn init_hidden(&self, batch_size: usize, device: &Device) -> Result<Vec<MLstmstate>> {
        let head_dim = self.d_hidden / self.num_heads;
        (0..self.num_layers)
            .map(|_| {
                Ok(MLstmstate::new(
                    Tensor::zeros((batch_size, self.num_heads, head_dim, head_dim), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                ))
            })
            .collect()
    }
}

/// Cell implementation with RWKV-7 Delta Rule logic
#[derive(Debug)]
pub struct MLstmcell {
    pub w_k: Linear, // Key
    pub w_v: Linear, // Value
    pub w_r: Linear, // Receptance (like Output gate/Query)
    pub w_a: Linear, // Alpha (Learning Rate gate)
    pub w_w: Linear, // Decay rate (Time-decay)
    
    pub ln: LayerNorm,
    
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
}

impl MLstmcell {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // Projections
        // Note: Using linear_no_bias to match the matrix concept cleanly.
        let w_k = linear_no_bias(input_size, hidden_size, vb.pp("w_k"))?;
        let w_v = linear_no_bias(input_size, hidden_size, vb.pp("w_v"))?;
        let w_r = linear_no_bias(input_size, hidden_size, vb.pp("w_r"))?;
        let w_a = linear_no_bias(input_size, hidden_size, vb.pp("w_a"))?;
        let w_w = linear_no_bias(input_size, hidden_size, vb.pp("w_w"))?;

        let ln = layer_norm(head_dim, 1e-5, vb.pp("ln"))?;

        Ok(Self {
            w_k,
            w_v,
            w_r,
            w_a,
            w_w,
            ln,
            input_size,
            hidden_size,
            num_heads,
        })
    }

    /// Forward pass processing the sequence step-by-step to apply the Delta Rule
    pub fn forward_sequence(
        &self,
        input_seq: &Tensor,
        state: &MLstmstate,
    ) -> Result<(Tensor, MLstmstate)> {
        let (batch_size, seq_len, _) = input_seq.dims3()?;
        let head_dim = self.hidden_size / self.num_heads;

        // 1. Pre-calculate all projections for speed (Parallel Projections)
        // [B, S, D] -> [B, S, H, D_head]
        let k_seq = self.project(input_seq, &self.w_k, batch_size, seq_len, head_dim)?;
        let v_seq = self.project(input_seq, &self.w_v, batch_size, seq_len, head_dim)?;
        let r_seq = self.project(input_seq, &self.w_r, batch_size, seq_len, head_dim)?;
        
        // Alpha (gate): Use Sigmoid as per request
        let a_seq_pre = self.project(input_seq, &self.w_a, batch_size, seq_len, head_dim)?;
        let a_seq = ops::sigmoid(&a_seq_pre)?;

        // Decay (w): Use Sigmoid to ensure 0..1 stability
        let w_seq_pre = self.project(input_seq, &self.w_w, batch_size, seq_len, head_dim)?;
        let w_seq = ops::sigmoid(&w_seq_pre)?;
        
        // Stabilize K: Scale by 1/sqrt(head_dim) to prevent update explosion
        // This acts as normalizing the "step size" of the delta rule.
        let k_seq = (k_seq / (head_dim as f64).sqrt())?;

        // 2. Sequential Scan (Delta Rule)
        let mut current_cell = state.cell.clone(); // [B, H, D, D]
        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Slice timestep t: [B, H, 1, D]
            let k_t = k_seq.narrow(2, t, 1)?; // [B, H, 1, D]
            let v_t = v_seq.narrow(2, t, 1)?; // [B, H, 1, D]
            let r_t = r_seq.narrow(2, t, 1)?; // [B, H, 1, D]
            let a_t = a_seq.narrow(2, t, 1)?; // [B, H, 1, D]
            let w_t = w_seq.narrow(2, t, 1)?; // [B, H, 1, D]

            // Reshape for matrix ops
            // k_t: [B, H, 1, D]
            // current_cell: [B, H, D, D]
            
            // Prediction = State * k
            // [B, H, D, D] matmul [B, H, D, 1] (k_t transposed) -> [B, H, D, 1]
            // Wait, k_t is [1, D]. We want State @ k^T.
            // Let's align dimensions accurately.
            // k_t_vec: [B, H, D, 1]
            let k_t_vec = k_t.permute((0, 1, 3, 2))?; // [B, H, D, 1]
            let prediction = current_cell.matmul(&k_t_vec)?; // [B, H, D, 1]

            // Delta = v - prediction
            // v_t is [B, H, 1, D]. We need [B, H, D, 1] for subtraction or transpose v.
            let v_t_vec = v_t.permute((0, 1, 3, 2))?; // [B, H, D, 1]
            let delta = (v_t_vec - &prediction)?; // [B, H, D, 1]
            
            // Update State
            // 1. Decay: state = state * w_t
            // w_t is [B, H, 1, D]. We want to broadcast to [B, H, D, D].
            // Usually decay applies to the memory rows? Or Uniform?
            // User: "self.state *= &w".
            // If w is per-channel D, we broadcast [1, D] to [D, D] (columns) or [D, 1] (rows).
            // RWKV typically decays keys. 
            // We'll broadcast w_t [B, H, 1, D] to [B, H, D, D] (columns decayed).
            // Or w_t_vec [B, H, D, 1].
            // Let's decay the whole matrix elementwise via broadcast.
            current_cell = current_cell.broadcast_mul(&w_t)?;

            // 2. Add: state += delta @ k
            // delta: [B, H, D, 1]
            // k_t: [B, H, 1, D]
            // delta @ k_t -> [B, H, D, D]
            // Modulated by alpha 'a'.
            // User: update_state_with_delta(&delta, &k, &a)
            // Implicitly: delta @ k * a?
            // "v - S*K" is a vector. "K" is a vector. Outer product.
            // (v - S u) \otimes v ? No (v - S k) \otimes k.
            // alpha scales the update speed.
            let update_term = delta.matmul(&k_t)?; // [B, H, D, D]
            // Apply alpha scaling. a_t: [B, H, 1, D]. Broadcast to [B, H, D, D]?
            let update_weighted = update_term.broadcast_mul(&a_t)?;
            
            current_cell = (current_cell + update_weighted)?;

            // Output Generation
            // r * (State * k)
            // Note: User code calculates `r * self.state.dot(&k)`.
            // Does it use the *updated* state or old state?
            // Usually output comes from updated state (current timestep view).
            // "Standard" RNN: h_t = f(x_t, h_{t-1}). Output y_t = g(h_t).
            // So we use current_cell (updated).
            
            // Re-calculate projection with new state
            // prediction_new = State_new * k
            let pred_new = current_cell.matmul(&k_t_vec)?; // [B, H, D, 1]
            
            // Apply Receptance r
            // r_t_vec: [B, H, D, 1]
            let r_t_vec = r_t.permute((0, 1, 3, 2))?;
            let out_vec = r_t_vec.broadcast_mul(&pred_new)?; // [B, H, D, 1]

            // Norm? User didn't specify, but mLSTM/RWKV usually expects LayerNorm on the output vector.
            // Flatten to [B, H, D]
            let out_flat = out_vec.squeeze(3)?; 
            outputs.push(out_flat);
        }

        // Stack outputs: [B, S, H, D]
        let seq_tensor = Tensor::stack(&outputs, 1)?; // [B, S, H, D_head]
        
        // Apply LayerNorm
        // ln expects [..., D]. 
        let ln_out = self.ln.forward(&seq_tensor)?;

        // Reshape to [B, S, Hidden]
        let final_out = ln_out.flatten_from(2)?;

        // Final state
        // last output hidden state for 'hidden' field
        let last_hidden = final_out.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        Ok((final_out, MLstmstate::new(current_cell, last_hidden)))
    }

    fn project(
        &self, 
        input: &Tensor, 
        layer: &Linear, 
        b: usize, 
        s: usize, 
        h_dim: usize
    ) -> Result<Tensor> {
        // [B, S, H*D] -> [B, S, H, D] -> [B, H, S, D]
        layer.forward(input)?
            .reshape((b, s, self.num_heads, h_dim))?
            .permute((0, 2, 1, 3))?
            .contiguous()
    }
}
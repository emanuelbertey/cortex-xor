use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, Module, VarBuilder, Linear, LayerNorm, ops, linear_no_bias, layer_norm, init::Init};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct MLstmstate {
    pub cell: Tensor,
    pub hidden: Tensor,
    pub normalizer: Tensor,
    pub max_gate_log: Tensor,
}

impl MLstmstate {
    pub fn new(cell: Tensor, hidden: Tensor, normalizer: Tensor, max_gate_log: Tensor) -> Self {
        Self { cell, hidden, normalizer, max_gate_log }
    }

    pub fn detach(&self) -> Self {
        Self {
            cell: self.cell.detach(),
            hidden: self.hidden.detach(),
            normalizer: self.normalizer.detach(),
            max_gate_log: self.max_gate_log.detach(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLstmconfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub expansion_factor: usize,
    pub dropout: f32,
    pub weight_stdev: f64,
    pub forget_bias: f32,
    pub input_gate_bias: f32,
    pub output_gate_bias: f32,
    pub epsilon: f64,
    pub log_clamp: f64,
    pub exp_clamp: f64,
    pub norm_clamp_min: f64,
    pub norm_clamp_max: f64,
    pub exp_gate_scale: f64,
}

impl MLstmconfig {
    pub fn new(d_input: usize, d_hidden: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            d_input, d_hidden, num_layers, num_heads,
            expansion_factor: 2,
            dropout: 0.0,
            weight_stdev: 0.02,
            forget_bias: 1.0,
            input_gate_bias: 0.0,
            output_gate_bias: 0.0,
            epsilon: 1e-6,
            log_clamp: 1e-8,
            exp_clamp: 15.0,
            norm_clamp_min: 1e-6,
            norm_clamp_max: 1e10,
            exp_gate_scale: 1.0,
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
            layers.push(MLstmcell::new(input_size, self.d_hidden, self.num_heads, self.expansion_factor, self, vb.pp(format!("layer_{}", i)))?);
        }
        Ok(MLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            config: self.clone(),
        })
    }
}

#[derive(Debug)]
pub struct MLstm {
    pub layers: Vec<MLstmcell>,
    pub dropout_layer: Dropout,
    pub config: MLstmconfig,
}

impl MLstm {
    pub fn forward(&self, input_seq: &Tensor, states: Option<Vec<MLstmstate>>) -> Result<(Tensor, Vec<MLstmstate>)> {
        let (batch_size, _, _) = input_seq.dims3()?;
        let mut hidden_states = match states {
            Some(s) => s,
            None => self.init_hidden(batch_size, input_seq.device())?,
        };
        
        let mut layer_input = input_seq.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let (h_seq, new_state) = layer.forward_sequence(&layer_input, &hidden_states[i])?;
            hidden_states[i] = new_state;
            layer_input = h_seq;
        }
        Ok((layer_input, hidden_states))
    }

    fn init_hidden(&self, batch_size: usize, device: &Device) -> Result<Vec<MLstmstate>> {
        let d_inner = self.config.d_hidden * self.config.expansion_factor;
        let head_dim = d_inner / self.config.num_heads;
        (0..self.config.num_layers).map(|_| {
            Ok(MLstmstate::new(
                Tensor::zeros((batch_size, self.config.num_heads, head_dim, head_dim), DType::F32, device)?,
                Tensor::zeros((batch_size, self.config.d_hidden), DType::F32, device)?,
                Tensor::zeros((batch_size, self.config.num_heads, head_dim), DType::F32, device)?,
                Tensor::zeros((batch_size, self.config.num_heads, 1), DType::F32, device)?,
            ))
        }).collect()
    }
}

#[derive(Debug)]
pub struct MLstmcell {
    pub w_gates: Linear,
    pub gate_bias: Tensor,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_down: Linear,
    pub ln: LayerNorm,
    pub config: MLstmconfig,
}

impl MLstmcell {
    pub fn new(input_size: usize, hidden_size: usize, num_heads: usize, expansion_factor: usize, config: &MLstmconfig, vb: VarBuilder) -> Result<Self> {
        let d_inner = hidden_size * expansion_factor;
        let head_dim = d_inner / num_heads;
        
        let w_gates = linear_no_bias(input_size, 3 * d_inner, vb.pp("w_gates"))?;
        
        // Inicialización manual de bias para inyectar en el VarBuilder
        let mut b_vals = vec![0.0f32; 3 * d_inner];
        for i in 0..d_inner { b_vals[i] = config.input_gate_bias; }
        for i in d_inner..(2 * d_inner) { b_vals[i] = config.forget_bias; }
        for i in (2 * d_inner)..(3 * d_inner) { b_vals[i] = config.output_gate_bias; }
        
        let gate_bias = vb.get_with_hints(
            (3 * d_inner,), 
            "gate_bias", 
            Init::Const(0.0)
        )?;
        
        let w_q = linear_no_bias(input_size, d_inner, vb.pp("w_q"))?;
        let w_k = linear_no_bias(input_size, d_inner, vb.pp("w_k"))?;
        let w_v = linear_no_bias(input_size, d_inner, vb.pp("w_v"))?;
        let w_down = linear_no_bias(d_inner, hidden_size, vb.pp("w_down"))?;
        let ln = layer_norm(head_dim, 1e-5, vb.pp("ln"))?;

        Ok(Self { w_gates, gate_bias, w_q, w_k, w_v, w_down, ln, config: config.clone() })
    }

    pub fn forward_sequence(&self, input_seq: &Tensor, state: &MLstmstate) -> Result<(Tensor, MLstmstate)> {
        let (batch_size, seq_len, _) = input_seq.dims3()?;
        let d_inner = self.config.d_hidden * self.config.expansion_factor;
        let head_dim = d_inner / self.config.num_heads;
        let device = input_seq.device();

        let q = (self.w_q.forward(input_seq)? / (head_dim as f64).sqrt())?
            .reshape((batch_size, seq_len, self.config.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = (self.w_k.forward(input_seq)? / (head_dim as f64).sqrt())?
            .reshape((batch_size, seq_len, self.config.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = self.w_v.forward(input_seq)?
            .reshape((batch_size, seq_len, self.config.num_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

        let gates = self.w_gates.forward(input_seq)?.broadcast_add(&self.gate_bias)?;
        let chunks = gates.chunk(3, 2)?;
        
        let i_gate = (chunks[0].reshape((batch_size, seq_len, self.config.num_heads, head_dim))?.permute((0, 2, 1, 3))? / self.config.exp_gate_scale)?
            .clamp(-self.config.exp_clamp, self.config.exp_clamp)?.exp()?;
        let f_gate = ops::sigmoid(&chunks[1].reshape((batch_size, seq_len, self.config.num_heads, head_dim))?.permute((0, 2, 1, 3))?)?;
        let o_gate = ops::sigmoid(&chunks[2])?;

        let i_log_m = i_gate.clamp(self.config.log_clamp as f32, f32::MAX)?.log()?.mean(3)?.unsqueeze(3)?;
        let f_log_m = f_gate.clamp(self.config.log_clamp as f32, 1.0)?.log()?.mean(3)?.unsqueeze(3)?;

        // Máscara Causal Nativa Candle
        let indices = Tensor::arange(0u32, seq_len as u32, device)?;
        let i_idx = indices.reshape((seq_len, 1))?.broadcast_as((seq_len, seq_len))?;
        let j_idx = indices.reshape((1, seq_len))?.broadcast_as((seq_len, seq_len))?;
        let mask_tri = i_idx.ge(&j_idx)?.to_dtype(DType::F32)?;

        let f_cumsum = mask_tri.broadcast_as((batch_size, self.config.num_heads, seq_len, seq_len))?.matmul(&f_log_m)?;
        let log_weights = f_cumsum.broadcast_sub(&f_cumsum.permute((0, 1, 3, 2))?)?.broadcast_add(&i_log_m.permute((0, 1, 3, 2))?)?;

        let m_0 = state.max_gate_log.reshape((batch_size, self.config.num_heads, 1, 1))?;
        let m_i = log_weights.max(3)?.unsqueeze(3)?.maximum(&f_cumsum.broadcast_add(&m_0)?)?;
        let weights = log_weights.broadcast_sub(&m_i)?.clamp(-20.0, 0.0)?.exp()?;

        let h_parallel = weights.matmul(&q.matmul(&k.t()?)?)?.matmul(&v)?;
        let initial_scale = f_cumsum.broadcast_add(&m_0)?.broadcast_sub(&m_i)?.exp()?;
        let h_initial = q.matmul(&state.cell)?.broadcast_mul(&initial_scale)?;
        
        let n_parallel = weights.matmul(&k.contiguous()?)?;
        let n_initial = state.normalizer.reshape((batch_size, self.config.num_heads, 1, head_dim))?.broadcast_mul(&initial_scale)?;
        let n_heads = (n_parallel + n_initial)?.clamp(self.config.norm_clamp_min as f32, f32::MAX)?;

        let h_normalized = self.ln.forward(&(h_parallel + h_initial)?.div(&n_heads)?.permute((0, 2, 1, 3))?.contiguous()?)?;
        let h_seq = self.w_down.forward(&(h_normalized.reshape((batch_size, seq_len, d_inner))? * o_gate)?)?;

        let last_idx = seq_len - 1;
        let final_m = m_i.narrow(2, last_idx, 1)?.reshape((batch_size, self.config.num_heads, 1))?;
        let final_norm = n_heads.narrow(2, last_idx, 1)?.reshape((batch_size, self.config.num_heads, head_dim))?;
        let final_cell = (state.cell.broadcast_mul(&initial_scale.narrow(2, last_idx, 1)?)? + v.narrow(2, last_idx, 1)?.permute((0, 1, 3, 2))?.matmul(&k.narrow(2, last_idx, 1)?)?)?;
        let final_hidden = h_seq.narrow(1, last_idx, 1)?.reshape((batch_size, self.config.d_hidden))?;

        Ok((h_seq, MLstmstate::new(final_cell, final_hidden, final_norm, final_m)))
    }
}
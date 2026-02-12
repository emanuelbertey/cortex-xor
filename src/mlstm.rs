/*!
# mLSTM: Matrix Long Short-Term Memory (Candle)
Implementación definitiva con Gates Exponenciales y Estabilización Numérica.
Variables de ajuste de estabilidad configurables (bias, dropout, clamp, epsilon, etc.).
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, VarBuilder, ops};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct MLstmstate {
    pub cell: Tensor,         // [B, Hh, Hd, Hd]
    pub hidden: Tensor,       // [B, H]
    pub normalizer: Tensor,   // [B, Hh, Hd]
    pub max_gate_log: Tensor, // [B, Hh, 1]
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
    pub dropout: f32,
    // --- Variables de ajuste de estabilidad ---
    /// Desviación estándar para inicialización de pesos (Kaiming o Normal).
    pub weight_stdev: f64,
    /// Bias aplicado al forget gate para controlar la retención de memoria.
    pub forget_bias: f32,
    /// Bias aplicado al input gate exponencial para controlar velocidad de escritura.
    pub input_gate_bias: f32,
    /// Epsilon para estabilización numérica en la normalización.
    pub epsilon: f64,
    /// Clamp inferior para log-gates (evita log(0)).
    pub log_clamp: f64,
    /// Clamp máximo para exponenciales (evita overflow en exp gates).
    pub exp_clamp: f64,
    /// Clamp para el denominador de normalización.
    pub norm_clamp_min: f64,
    pub norm_clamp_max: f64,
    /// Escala de los gates exponenciales (divide la pre-activación).
    pub exp_gate_scale: f64,
    /// Si true, usar bias separado por gate (input, forget, output).
    pub use_separate_bias: bool,
}

impl MLstmconfig {
    pub fn new(d_input: usize, d_hidden: usize, num_layers: usize, num_heads: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            num_layers,
            num_heads,
            dropout: 0.0,
            // Valores por defecto sugeridos para estabilidad
            weight_stdev: 0.02,
            forget_bias: 0.5,
            input_gate_bias: 0.0,
            epsilon: 1e-6,
            log_clamp: 1e-4,
            exp_clamp: 20.0,
            norm_clamp_min: 1e-6,
            norm_clamp_max: 1e10,
            exp_gate_scale: 2.0,
            use_separate_bias: true,
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Configura los parámetros de estabilidad numérica principales.
    pub fn with_stability(mut self, weight_stdev: f64, forget_bias: f32, epsilon: f64) -> Self {
        self.weight_stdev = weight_stdev;
        self.forget_bias = forget_bias;
        self.epsilon = epsilon;
        self
    }

    /// Configura los clamps para gates exponenciales y normalización.
    pub fn with_clamps(mut self, log_clamp: f64, exp_clamp: f64, norm_min: f64, norm_max: f64) -> Self {
        self.log_clamp = log_clamp;
        self.exp_clamp = exp_clamp;
        self.norm_clamp_min = norm_min;
        self.norm_clamp_max = norm_max;
        self
    }

    /// Configura la escala del gate exponencial.
    pub fn with_exp_gate_scale(mut self, scale: f64) -> Self {
        self.exp_gate_scale = scale;
        self
    }

    /// Configura el bias del input gate exponencial.
    pub fn with_input_gate_bias(mut self, bias: f32) -> Self {
        self.input_gate_bias = bias;
        self
    }

    pub fn init(&self, vb: VarBuilder) -> Result<MLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            let layer_vb = vb.pp(format!("layer_{}", i));
            layers.push(MLstmcell::new(input_size, self.d_hidden, self.num_heads, self, layer_vb)?);
        }
        Ok(MLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            dropout: self.dropout,
        })
    }
}

#[derive(Debug)]
pub struct MLstm {
    pub layers: Vec<MLstmcell>,
    pub dropout_layer: Dropout,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub dropout: f32,
}

impl MLstm {
    pub fn forward(&self, input_seq: &Tensor, states: Option<Vec<MLstmstate>>) -> Result<(Tensor, Vec<MLstmstate>)> {
        let (b, seq_len, _) = input_seq.dims3()?;
        if seq_len > 1 {
            let mut x = input_seq.clone();
            let mut final_states = Vec::new();
            for (i, layer) in self.layers.iter().enumerate() {
                let (out, last_s) = layer.forward_dual(&x)?;
                final_states.push(last_s.detach());
                x = out;
                if i < self.num_layers - 1 && self.dropout > 0.0 {
                    x = self.dropout_layer.forward(&x, true)?;
                }
            }
            Ok((x, final_states))
        } else {
            let device = input_seq.device();
            let mut hidden_states = match states {
                Some(s) => s,
                None => self.init_hidden(b, device)?,
            };
            let mut layer_input = input_seq.squeeze(1)?;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (h_new, new_state) = layer.forward_step(&layer_input, &hidden_states[layer_idx])?;
                hidden_states[layer_idx] = new_state.detach();
                layer_input = h_new;
            }
            Ok((layer_input.unsqueeze(1)?, hidden_states))
        }
    }

    pub fn init_hidden(&self, batch_size: usize, device: &Device) -> Result<Vec<MLstmstate>> {
        let head_dim = self.d_hidden / self.layers[0].num_heads;
        (0..self.num_layers).map(|_| {
            Ok(MLstmstate::new(
                Tensor::zeros((batch_size, self.layers[0].num_heads, head_dim, head_dim), DType::F32, device)?,
                Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                Tensor::zeros((batch_size, self.layers[0].num_heads, head_dim), DType::F32, device)?,
                Tensor::zeros((batch_size, self.layers[0].num_heads, 1), DType::F32, device)?,
            ))
        }).collect()
    }
}

#[derive(Debug)]
pub struct MLstmcell {
    pub weight_ih: Tensor,
    pub bias: Tensor,
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    // Parámetros de ajuste almacenados
    pub forget_bias: f32,
    pub input_gate_bias: f32,
    pub epsilon: f64,
    pub log_clamp: f64,
    pub exp_clamp: f64,
    pub norm_clamp_min: f64,
    pub norm_clamp_max: f64,
    pub exp_gate_scale: f64,
}

impl MLstmcell {
    pub fn new(input_size: usize, hidden_size: usize, num_heads: usize, config: &MLstmconfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        let weight_init = candle_nn::init::Init::Randn {
            mean: 0.0,
            stdev: config.weight_stdev,
        };
        // xLSTM mLSTM necesita 6 proyecciones: q, k, v, i (input gate), f (forget gate), o (output gate)
        let weight_ih = vb.get_with_hints((6 * hidden_size, input_size), "weight_ih", weight_init)?;

        // Construir bias con valores diferenciados por gate si use_separate_bias
        let bias = if config.use_separate_bias {
            // [q_bias(0) | k_bias(0) | v_bias(0) | i_bias(input_gate_bias) | f_bias(forget_bias) | o_bias(0)]
            let mut bias_vals = vec![0.0f32; 6 * hidden_size];
            // chunk 3: input gate bias
            for i in (3 * hidden_size)..(4 * hidden_size) {
                bias_vals[i] = config.input_gate_bias;
            }
            // chunk 4: forget gate bias
            for i in (4 * hidden_size)..(5 * hidden_size) {
                bias_vals[i] = config.forget_bias;
            }
            let bias_tensor = Tensor::from_vec(bias_vals, (6 * hidden_size,), weight_ih.device())?;
            let bias_param = vb.get_with_hints(6 * hidden_size, "bias", candle_nn::init::Init::Const(0.0))?;
            bias_param.broadcast_add(&bias_tensor)?
        } else {
            vb.get_with_hints(6 * hidden_size, "bias", candle_nn::init::Init::Const(0.0))?
        };

        Ok(Self {
            weight_ih,
            bias,
            num_heads,
            head_dim,
            hidden_size,
            forget_bias: config.forget_bias,
            input_gate_bias: config.input_gate_bias,
            epsilon: config.epsilon,
            log_clamp: config.log_clamp,
            exp_clamp: config.exp_clamp,
            norm_clamp_min: config.norm_clamp_min,
            norm_clamp_max: config.norm_clamp_max,
            exp_gate_scale: config.exp_gate_scale,
        })
    }

    pub fn forward_dual(&self, x: &Tensor) -> Result<(Tensor, MLstmstate)> {
        let (b, s, d) = x.dims3()?;
        let device = x.device();
        
        let x_flat = x.reshape((b * s, d))?;
        let all_gates_flat = x_flat.matmul(&self.weight_ih.t()?)?.broadcast_add(&self.bias)?;
        let all_gates = all_gates_flat.reshape((b, s, 6 * self.hidden_size))?;
        let chunks = all_gates.chunk(6, 2)?;
        
        let q = chunks[0].reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = chunks[1].reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = chunks[2].reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        // chunks[3] = i (input gate) - no usado en forward_dual (modo attention)
        // chunks[4] = f (forget gate) - no usado en forward_dual
        let o = ops::sigmoid(&chunks[5])?; // output gate

        // Scaled dot-product para estabilizar
        let scale = (self.head_dim as f64).sqrt().recip();
        let q_scaled = (q * scale)?;
        
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q_scaled.matmul(&k_t)?;
        
        let mask = self.get_causal_mask(s, device)?;
        let attn = ops::softmax(&scores.broadcast_add(&mask)?, 3)?;
        
        let out_heads = attn.matmul(&v)?;
        let out = out_heads.transpose(1, 2)?.contiguous()?.reshape((b, s, self.hidden_size))?;
        let h_final = out.broadcast_mul(&o)?;

        let last_s = MLstmstate::new(
            Tensor::zeros((b, self.num_heads, self.head_dim, self.head_dim), DType::F32, device)?,
            h_final.narrow(1, s - 1, 1)?.squeeze(1)?,
            Tensor::zeros((b, self.num_heads, self.head_dim), DType::F32, device)?,
            Tensor::zeros((b, self.num_heads, 1), DType::F32, device)?,
        );
        Ok((h_final, last_s))
    }

    pub fn forward_step(&self, input: &Tensor, state: &MLstmstate) -> Result<(Tensor, MLstmstate)> {
        let (b, _) = input.dims2()?;
        let device = input.device();
        let gates = input.matmul(&self.weight_ih.t()?)?.broadcast_add(&self.bias)?;
        let chunks = gates.chunk(6, 1)?;
        
        // xLSTM mLSTM: proyecciones separadas para q, k, v, i, f, o
        let q = chunks[0].reshape((b, self.num_heads, self.head_dim))?;
        let k = chunks[1].reshape((b, self.num_heads, self.head_dim))?;
        let v = chunks[2].reshape((b, self.num_heads, self.head_dim))?;
        let i_raw = chunks[3].reshape((b, self.num_heads, self.head_dim))?;
        let f_raw = chunks[4].reshape((b, self.num_heads, self.head_dim))?;
        let o = ops::sigmoid(&chunks[5])?;

        // 1. Input Gate Exponencial (Clave del xLSTM)
        // Escalamos por exp_gate_scale para controlar la magnitud, con clamp para estabilidad
        let i_scaled = (i_raw / self.exp_gate_scale)?;
        let i_clamped = i_scaled.clamp(-self.exp_clamp, self.exp_clamp)?;
        let i_gate = i_clamped.exp()?;

        // 2. Forget Gate sigmoidal (con bias configurable ya aplicado en la inicialización)
        let f_gate = ops::sigmoid(&f_raw)?;

        // 3. Actualización de Matriz de Memoria: C_t = f_t ⊙ C_{t-1} + i_t ⊙ (v ⊗ k)
        let mat_update = v.unsqueeze(3)?.matmul(&k.unsqueeze(2)?)?;

        let c_new = state.cell
            .broadcast_mul(&f_gate.unsqueeze(3)?)?
            .add(&mat_update.broadcast_mul(&i_gate.unsqueeze(3)?)?)?;

        // 4. Normalizador: n_t = f_t ⊙ n_{t-1} + i_t ⊙ k
        let n_new = state.normalizer.broadcast_mul(&f_gate)?
            .add(&k.broadcast_mul(&i_gate)?)?;

        // 5. Retrieval: h̃_t = C_t @ q
        let h_raw = c_new.matmul(&q.unsqueeze(3)?)?.squeeze(3)?;
        
        // 6. Normalización estabilizada: h_t = h̃_t / max(|n_t · q|, ε)
        let dot_prod = n_new.broadcast_mul(&q)?.sum_keepdim(2)?;
        let eps_tensor = Tensor::new(&[self.epsilon as f32], device)?;
        let den = dot_prod.abs()?.broadcast_add(&eps_tensor)?;
        let den = den.clamp(self.norm_clamp_min, self.norm_clamp_max)?;
        let h_norm = h_raw.broadcast_div(&den)?;
        
        // 7. Output gate: h_final = o_t ⊙ h_t
        let h_final = h_norm.reshape((b, self.hidden_size))?.broadcast_mul(&o)?;

        Ok((h_final.clone(), MLstmstate::new(c_new, h_final, n_new, state.max_gate_log.clone())))
    }

    fn get_causal_mask(&self, s: usize, device: &Device) -> Result<Tensor> {
        let mask: Vec<f32> = (0..s)
            .flat_map(|i| (0..s).map(move |j| if j <= i { 0.0 } else { -1e9 }))
            .collect();
        Tensor::from_vec(mask, (1, 1, s, s), device)
    }
}

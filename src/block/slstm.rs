/*!
# sLSTM: Scalar Long Short-Term Memory
Implementation according to: "xLSTM: Extended Long Short-Term Memory" (2405.04517v2)

The sLSTM (Scalar LSTM) extends the traditional LSTM with:
1. Exponential gating with a log-space max-stabilizer (m_t).
2. Normalization (n_t) of the cell state to maintain boundedness.
3. Memory mixing via the hidden-to-state recurrent connection.
*/

use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Dropout, VarBuilder, ops};

/// State for sLSTM containing cell, normalizer, hidden, and stabilizer states
#[derive(Clone, Debug)]
pub struct SLstmstate {
    /// Cell state (c_t)
    pub cell: Tensor,
    /// Normalizer state (n_t)
    pub normalizer: Tensor,
    /// Hidden state (h_t)
    pub hidden: Tensor,
    /// Stabilizer state (m_t: tracking the maximum gate logit)
    pub stabilizer: Tensor,
}

impl SLstmstate {
    pub fn new(
        cell: Tensor,
        normalizer: Tensor,
        hidden: Tensor,
        stabilizer: Tensor,
    ) -> Self {
        Self {
            cell,
            normalizer,
            hidden,
            stabilizer,
        }
    }

    pub fn detach(&self) -> Self {
        Self {
            cell: self.cell.detach(),
            normalizer: self.normalizer.detach(),
            hidden: self.hidden.detach(),
            stabilizer: self.stabilizer.detach(),
        }
    }
}

/// Configuration for sLSTM
#[derive(Debug, Clone)]
pub struct SLstmconfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub dropout: f32,
    // --- Variables de ajuste de estabilidad ---
    /// Desviación estándar para inicializ ación de pesos.
    pub weight_stdev: f64,
    /// Bias para forget gate (logit).
    pub forget_bias: f32,
    /// Bias para input gate (logit).
    pub input_bias: f32,
    /// Epsilon para evitar división por cero en normalización.
    pub epsilon: f64,
    /// Clamp mínimo para exponenciales estabilizadas (evitar underflow).
    pub exp_clamp_min: f64,
    /// Clamp máximo para exponenciales (evitar overflow).
    pub exp_clamp_max: f64,
    /// Valor inicial del estabilizador m_0.
    pub stabilizer_init: f32,
    /// Si true, aplicar bias diferenciados por gate.
    pub use_separate_bias: bool,
}

impl SLstmconfig {
    pub fn new(d_input: usize, d_hidden: usize, num_layers: usize) -> Self {
        Self {
            d_input,
            d_hidden,
            num_layers,
            dropout: 0.0,
            // Valores por defecto para estabilidad
            weight_stdev: 0.02,
            forget_bias: 0.0,
            input_bias: 0.0,
            epsilon: 1e-6,
            exp_clamp_min: -30.0,
            exp_clamp_max: 0.0,
            stabilizer_init: -10.0,
            use_separate_bias: true,
        }
    }

    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Configura parámetros de estabilidad numérica.
    pub fn with_stability(mut self, weight_stdev: f64, forget_bias: f32, epsilon: f64) -> Self {
        self.weight_stdev = weight_stdev;
        self.forget_bias = forget_bias;
        self.epsilon = epsilon;
        self
    }

    /// Configura clamps para exponenciales.
    pub fn with_exp_clamps(mut self, min: f64, max: f64) -> Self {
        self.exp_clamp_min = min;
        self.exp_clamp_max = max;
        self
    }

    /// Configura valor inicial del estabilizador.
    pub fn with_stabilizer_init(mut self, val: f32) -> Self {
        self.stabilizer_init = val;
        self
    }

    pub fn init(&self, vb: VarBuilder) -> Result<SLstm> {
        let mut layers = Vec::with_capacity(self.num_layers);
        for i in 0..self.num_layers {
            let input_size = if i == 0 { self.d_input } else { self.d_hidden };
            let layer_vb = vb.pp(format!("layer_{}", i));
            layers.push(SLstmcell::new(input_size, self.d_hidden, self, layer_vb)?);
        }

        Ok(SLstm {
            layers,
            dropout_layer: Dropout::new(self.dropout),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            dropout: self.dropout,
            stabilizer_init: self.stabilizer_init,
        })
    }
}

/// sLSTM layer stack implementation
#[derive(Debug)]
pub struct SLstm {
    pub layers: Vec<SLstmcell>,
    pub dropout_layer: Dropout,
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub stabilizer_init: f32,
}

impl SLstm {
    pub fn forward(
        &self,
        input_seq: &Tensor,
        states: Option<Vec<SLstmstate>>,
    ) -> Result<(Tensor, Vec<SLstmstate>)> {
        let (batch_size, seq_length, _) = input_seq.dims3()?;
        let device = input_seq.device();

        let mut hidden_states = match states {
            Some(s) => s,
            None => self.init_hidden(batch_size, device)?,
        };

        let mut all_outputs = Vec::with_capacity(seq_length);

        for t in 0..seq_length {
            let input_t = input_seq.narrow(1, t, 1)?.squeeze(1)?;
            let mut layer_input = input_t;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let (h_new, new_state) = layer.forward(&layer_input, &hidden_states[layer_idx])?;
                hidden_states[layer_idx] = new_state;

                layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                    self.dropout_layer.forward(&h_new, true)? 
                } else {
                    h_new
                };
            }

            all_outputs.push(layer_input.unsqueeze(1)?);
        }

        let output = Tensor::cat(&all_outputs, 1)?;
        Ok((output, hidden_states))
    }

    fn init_hidden(&self, batch_size: usize, device: &Device) -> Result<Vec<SLstmstate>> {
        (0..self.num_layers)
            .map(|_| {
                Ok(SLstmstate::new(
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    // xLSTM paper: n_0 = 0 (Equation 13 & 14 starting sum from 0)
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    Tensor::zeros((batch_size, self.d_hidden), DType::F32, device)?,
                    // m_0 configurable (permite ajuste fino)
                    (Tensor::ones((batch_size, self.d_hidden), DType::F32, device)? * (self.stabilizer_init as f64))?,
                ))
            })
            .collect()
    }
}

/// sLSTM cell implementation with Exponential Gating (Eq. 15-20)
#[derive(Debug)]
pub struct SLstmcell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias: Tensor,
    pub input_size: usize,
    pub hidden_size: usize,
    // Parámetros de ajuste
    pub forget_bias: f32,
    pub input_bias: f32,
    pub epsilon: f64,
    pub exp_clamp_min: f64,
    pub exp_clamp_max: f64,
}

impl SLstmcell {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        config: &SLstmconfig,
        vb: VarBuilder, 
    ) -> Result<Self> {
        // xLSTM paper suggests standard initialization for LSTM-like gates.
        let weight_init = candle_nn::init::Init::Randn {
            mean: 0.0,
            stdev: config.weight_stdev,
        };
        
        let weight_ih = vb.get_with_hints(
            (4 * hidden_size, input_size), 
            "weight_ih", 
            weight_init
        )?;
        
        let weight_hh = vb.get_with_hints(
            (4 * hidden_size, hidden_size), 
            "weight_hh", 
            weight_init
        )?;

        // Bias con valores diferenciados si use_separate_bias
        let bias = if config.use_separate_bias {
            // [i_bias | f_bias | z_bias(0) | o_bias(0)]
            let mut bias_vals = vec![0.0f32; 4 * hidden_size];
            for i in 0..hidden_size {
                bias_vals[i] = config.input_bias;
            }
            for i in hidden_size..(2 * hidden_size) {
                bias_vals[i] = config.forget_bias;
            }
            let bias_tensor = Tensor::from_vec(bias_vals, (4 * hidden_size,), weight_ih.device())?;
            let bias_param = vb.get_with_hints(4 * hidden_size, "bias", candle_nn::init::Init::Const(0.0))?;
            bias_param.broadcast_add(&bias_tensor)?
        } else {
            vb.get_with_hints(4 * hidden_size, "bias", candle_nn::init::Init::Const(0.0))?
        };

        Ok(Self {
            weight_ih,
            weight_hh,
            bias,
            input_size,
            hidden_size,
            forget_bias: config.forget_bias,
            input_bias: config.input_bias,
            epsilon: config.epsilon,
            exp_clamp_min: config.exp_clamp_min,
            exp_clamp_max: config.exp_clamp_max,
        })
    }

    /// Forward pass through sLSTM cell (Single Step)
    pub fn forward(
        &self,
        input: &Tensor,
        state: &SLstmstate,
    ) -> Result<(Tensor, SLstmstate)> {
        let SLstmstate {
            cell,
            normalizer,
            hidden,
            stabilizer,
        } = state;

        // 1. Projections for i, f, z, o gates [B, 4*H]
        let gates = input.matmul(&self.weight_ih.t()?)?
            .broadcast_add(&self.bias.unsqueeze(0)?)?
            .broadcast_add(&hidden.matmul(&self.weight_hh.t()?)?)?;

        // 2. Gate separation (Assuming order: i, f, z, o)
        let chunks = gates.chunk(4, 1)?;
        let i_gate = &chunks[0]; // Logit del Input gate
        let f_gate = &chunks[1]; // Logit del Forget gate
        let z_gate = &chunks[2]; // Logit del Input content (phi)
        let o_gate = &chunks[3]; // Logit del Output gate (sigma)

        // 3. Log-Space Max Stabilization (Eq. 18-19)
        // m_t = max(f_gate + m_prev, i_gate)
        let m_prev_plus_f = stabilizer.add(f_gate)?;
        let m_new = m_prev_plus_f.maximum(i_gate)?; 

        // 4. Stabilized Exponential Gating (con clamps configurables)
        // i_t' = exp(i_gate - m_t)
        // f_t' = exp(f_gate + m_prev - m_t)
        let i_exp = (i_gate - &m_new)?.clamp(self.exp_clamp_min, self.exp_clamp_max)?.exp()?;
        let f_exp = (m_prev_plus_f - &m_new)?.clamp(self.exp_clamp_min, self.exp_clamp_max)?.exp()?;

        // Activaciones no-exponenciales
        let z = z_gate.tanh()?;      // Input content (z en el paper)
        let o = ops::sigmoid(o_gate)?; // Output gate (sigma)

        // 5. State Updates (Eq. 13-14)
        // c_t = f_t' * c_prev + i_t' * z_t
        // n_t = f_t' * n_prev + i_t'
        let c_new = ((&f_exp * cell)? + (&i_exp * z)?)?;
        let n_new = ((f_exp * normalizer)? + i_exp)?;

        // 6. Normalization and Output (Eq. 16-17) con epsilon configurable
        // Hidden state h_t = o_t * (c_t / n_t)
        // xLSTM paper: "The normalization ensures that h_t is bounded, making tanh unnecessary."
        let n_safe = n_new.clamp(self.epsilon, f32::MAX)?; 
        let h_new = (o * (c_new.clone() / n_safe)?)?;

        let new_state = SLstmstate::new(c_new, n_new, h_new.clone(), m_new);
        Ok((h_new, new_state))
    }
}

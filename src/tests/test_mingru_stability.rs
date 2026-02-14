use candle_core::{Tensor, Device, DType};
use candle_nn::{VarBuilder, VarMap};
use candle_nn::Optimizer;
use std::error::Error;

// Importar nuestro MinGRU
use xlstm::{MinGru, MinGruConfig};
use xlstm::{MinLstm, MinLstmConfig};

fn count_nan_inf(t: &Tensor) -> Result<(usize, usize), Box<dyn Error>> {
    let v = t.flatten_all()?.to_vec1::<f32>()?;
    let mut nan = 0usize;
    let mut inf = 0usize;
    for x in v {
        if x.is_nan() {
            nan += 1;
        }
        if x.is_infinite() {
            inf += 1;
        }
    }
    Ok((nan, inf))
}

struct ReferenceGru {
    w_x: Tensor,
    w_h: Tensor,
    b: Tensor,
    hidden_size: usize,
}

impl ReferenceGru {
    fn new(device: &Device, input_size: usize, hidden_size: usize) -> Result<Self, Box<dyn Error>> {
        let w_x = Tensor::randn(0.0f32, 0.02f32, (input_size, 3 * hidden_size), device)?;
        let w_h = Tensor::randn(0.0f32, 0.02f32, (hidden_size, 3 * hidden_size), device)?;
        let b = Tensor::zeros((3 * hidden_size,), DType::F32, device)?;
        Ok(Self { w_x, w_h, b, hidden_size })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let hsz = self.hidden_size;

        let mut h = Tensor::zeros((batch, hsz), DType::F32, x.device())?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq);

        let wx_zr = self.w_x.narrow(1, 0, 2 * hsz)?;
        let wh_zr = self.w_h.narrow(1, 0, 2 * hsz)?;
        let b_zr = self.b.narrow(0, 0, 2 * hsz)?;

        let wx_n = self.w_x.narrow(1, 2 * hsz, hsz)?;
        let wh_n = self.w_h.narrow(1, 2 * hsz, hsz)?;
        let b_n = self.b.narrow(0, 2 * hsz, hsz)?;

        for t in 0..seq {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?; // [b, in]

            let b_zr_bt = b_zr.unsqueeze(0)?.broadcast_as((batch, 2 * hsz))?;
            let gates_zr = x_t.matmul(&wx_zr)?.add(&h.matmul(&wh_zr)?)?.add(&b_zr_bt)?;
            let zr = gates_zr.chunk(2, 1)?;
            let z = candle_nn::ops::sigmoid(&zr[0])?;
            let r = candle_nn::ops::sigmoid(&zr[1])?;

            let b_n_bt = b_n.unsqueeze(0)?.broadcast_as((batch, hsz))?;
            let rh = r.mul(&h)?;
            let n_pre = x_t.matmul(&wx_n)?.add(&rh.matmul(&wh_n)?)?.add(&b_n_bt)?;
            let n = n_pre.tanh()?;

            let one_minus_z = Tensor::ones_like(&z)?.sub(&z)?;
            h = n.mul(&one_minus_z)?.add(&h.mul(&z)?)?;
            outputs.push(h.unsqueeze(1)?);
        }

        Ok(Tensor::cat(&outputs, 1)?)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Test de Estabilidad Numérica MinGRU ===");
    
    let device = Device::Cpu;
    
    // Crear configuración simple
    let config = MinGruConfig::new(256);
    
    // Inicializar MinGRU
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let mingru = config.init(vb)?;
    
    println!("✅ MinGRU inicializado correctamente");
    
    // Test 1: Datos aleatorios pequeños
    println!("\n--- Test 1: Datos aleatorios pequeños ---");
    test_random_data(&mingru, &device, 16, 32, 256)?;
    
    // Test 2: Datos secuenciales
    println!("\n--- Test 2: Datos secuenciales ---");
    test_sequential_data(&mingru, &device)?;
    
    // Test 3: Valores extremos
    println!("\n--- Test 3: Valores extremos ---");
    test_extreme_values(&mingru, &device)?;
    
    // Test 4: Forward paso a paso
    println!("\n--- Test 4: Forward paso a paso ---");
    test_step_by_step(&mingru, &device)?;

    // Test 5: Operaciones Candle (softmax/gelu)
    println!("\n--- Test 5: Operaciones Candle (softmax/gelu) ---");
    test_candle_ops(&device)?;

    println!("\n=== Test de Estabilidad Numérica MinLSTM ===");

    let minlstm_config = MinLstmConfig::new(256);
    let minlstm_varmap = VarMap::new();
    let minlstm_vb = VarBuilder::from_varmap(&minlstm_varmap, DType::F32, &device);
    let minlstm = minlstm_config.init(minlstm_vb)?;

    println!("✅ MinLSTM inicializado correctamente");

    println!("\n--- Test 6: MinLSTM datos aleatorios ---");
    test_random_data_minlstm(&minlstm, &device, 16, 32, 256)?;

    println!("\n--- Test 7: MinLSTM forward paso a paso ---");
    test_step_by_step_minlstm(&minlstm, &device)?;

    println!("\n--- Test 8: MinLSTM vs LSTM real (forward) ---");
    test_minlstm_vs_reference_lstm(&minlstm, &device)?;

    println!("\n--- Test 9: MinGRU vs GRU real (forward) ---");
    test_mingru_vs_reference_gru(&mingru, &device)?;

    println!("\n--- Test 10: Entrenamiento smoke-test + puntajes (MinLSTM vs MinGRU) ---");
    test_training_scores(&device)?;

    println!("\n--- Test 11: Tarea secuencial (suma acumulada) MinLSTM vs MinGRU ---");
    test_cumsum_task_scores(&device)?;
    
    println!("\n=== Todos los tests completados exitosamente! ===");
    Ok(())
}

fn test_mingru_vs_reference_gru(mingru: &MinGru, device: &Device) -> Result<(), Box<dyn Error>> {
    let batch_size = 4;
    let seq_len = 12;
    let hidden_size = 256;
    let x = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, hidden_size), device)?;

    let (out_mingru, _): (Tensor, Option<Tensor>) = mingru.forward(&x, None, true)?;
    let ref_gru = ReferenceGru::new(device, hidden_size, hidden_size)?;
    let out_gru = ref_gru.forward(&x)?;

    println!("MinGRU - Mean: {:.4}, Max: {:.4}",
        out_mingru.mean_all()?.to_scalar::<f32>()?,
        out_mingru.max_all()?.to_scalar::<f32>()?
    );
    println!("GRU real - Mean: {:.4}, Max: {:.4}",
        out_gru.mean_all()?.to_scalar::<f32>()?,
        out_gru.max_all()?.to_scalar::<f32>()?
    );

    Ok(())
}

fn test_random_data_minlstm(minlstm: &MinLstm, device: &Device, batch_size: usize, seq_len: usize, hidden_size: usize) -> Result<(), Box<dyn Error>> {
    let x = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, hidden_size), device)?;
    let (output, _): (Tensor, Option<Tensor>) = minlstm.forward(&x, None, true)?;

    println!("Output shape: {:?}", output.dims());
    println!("Output stats - Mean: {:.4}, Min: {:.4}, Max: {:.4}",
        output.mean_all()?.to_scalar::<f32>()?,
        output.min_all()?.to_scalar::<f32>()?,
        output.max_all()?.to_scalar::<f32>()?
    );

    let (nan_count, inf_count) = count_nan_inf(&output)?;
    println!("NaN count: {}, Inf count: {}", nan_count, inf_count);
    if nan_count > 0 || inf_count > 0 {
        return Err("NaN o Inf detectados en MinLSTM".into());
    }
    Ok(())
}

fn test_step_by_step_minlstm(minlstm: &MinLstm, device: &Device) -> Result<(), Box<dyn Error>> {
    let batch_size = 2;
    let hidden_size = 256;
    let seq_len = 5;

    let x = Tensor::randn(0.0f32, 0.1f32, (batch_size, seq_len, hidden_size), device)?;
    let output_full = minlstm.forward(&x, None, true)?.0;

    let mut current_state: Option<Tensor> = None;
    let mut step_outputs = Vec::new();
    for t in 0..seq_len {
        let step_input = x.narrow(1, t, 1)?.squeeze(1)?;
        let (step_output, next_state): (Tensor, Option<Tensor>) =
            minlstm.forward(&step_input.unsqueeze(1)?, current_state.as_ref(), true)?;
        current_state = next_state;
        step_outputs.push(step_output);
    }
    let output_steps = Tensor::cat(&step_outputs, 1)?;

    let diff = output_full.sub(&output_steps)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    println!("Diferencia máxima MinLSTM full vs step: {:.8}", max_diff);
    Ok(())
}

struct ReferenceLstm {
    w_ih: Tensor,
    w_hh: Tensor,
    b: Tensor,
    hidden_size: usize,
}

impl ReferenceLstm {
    fn new(device: &Device, input_size: usize, hidden_size: usize) -> Result<Self, Box<dyn Error>> {
        let w_ih = Tensor::randn(0.0f32, 0.02f32, (input_size, 4 * hidden_size), device)?;
        let w_hh = Tensor::randn(0.0f32, 0.02f32, (hidden_size, 4 * hidden_size), device)?;
        let b = Tensor::zeros((4 * hidden_size,), DType::F32, device)?;
        Ok(Self { w_ih, w_hh, b, hidden_size })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        let dims = x.dims();
        let batch = dims[0];
        let seq = dims[1];
        let mut h = Tensor::zeros((batch, self.hidden_size), DType::F32, x.device())?;
        let mut c = Tensor::zeros((batch, self.hidden_size), DType::F32, x.device())?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(seq);

        for t in 0..seq {
            let x_t = x.narrow(1, t, 1)?.squeeze(1)?; // [b, in]
            let b = self.b.unsqueeze(0)?.broadcast_as((batch, 4 * self.hidden_size))?;
            let gates = x_t.matmul(&self.w_ih)?.add(&h.matmul(&self.w_hh)?)?.add(&b)?;
            let chunks = gates.chunk(4, 1)?;
            let i = candle_nn::ops::sigmoid(&chunks[0])?;
            let f = candle_nn::ops::sigmoid(&chunks[1])?;
            let g = chunks[2].tanh()?;
            let o = candle_nn::ops::sigmoid(&chunks[3])?;
            c = f.mul(&c)?.add(&i.mul(&g)?)?;
            h = o.mul(&c.tanh()?)?;
            outputs.push(h.unsqueeze(1)?);
        }

        Ok(Tensor::cat(&outputs, 1)?)
    }
}

fn test_minlstm_vs_reference_lstm(minlstm: &MinLstm, device: &Device) -> Result<(), Box<dyn Error>> {
    let batch_size = 4;
    let seq_len = 12;
    let hidden_size = 256;
    let x = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, hidden_size), device)?;

    let (out_minlstm, _): (Tensor, Option<Tensor>) = minlstm.forward(&x, None, true)?;
    let ref_lstm = ReferenceLstm::new(device, hidden_size, hidden_size)?;
    let out_lstm = ref_lstm.forward(&x)?;

    println!("MinLSTM - Mean: {:.4}, Max: {:.4}",
        out_minlstm.mean_all()?.to_scalar::<f32>()?,
        out_minlstm.max_all()?.to_scalar::<f32>()?
    );
    println!("LSTM real - Mean: {:.4}, Max: {:.4}",
        out_lstm.mean_all()?.to_scalar::<f32>()?,
        out_lstm.max_all()?.to_scalar::<f32>()?
    );

    Ok(())
}

fn test_training_scores(device: &Device) -> Result<(), Box<dyn Error>> {
    use candle_nn::optim::AdamW;

    let batch_size = 8;
    let seq_len = 16;
    let dim = 64;
    let steps = 10;

    let x = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, dim), device)?;
    let scale = Tensor::full(0.5f32, x.shape(), device)?;
    let bias = Tensor::full(0.1f32, x.shape(), device)?;
    let target = x.mul(&scale)?.add(&bias)?;

    let varmap_gru = VarMap::new();
    let vb_gru = VarBuilder::from_varmap(&varmap_gru, DType::F32, device);
    let mingru = MinGruConfig::new(dim).init(vb_gru)?;
    let gru_params = {
        let data = varmap_gru.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    let mut optim_gru = AdamW::new_lr(gru_params, 5e-4)?;

    let varmap_lstm = VarMap::new();
    let vb_lstm = VarBuilder::from_varmap(&varmap_lstm, DType::F32, device);
    let minlstm = MinLstmConfig::new(dim).init(vb_lstm)?;
    let lstm_params = {
        let data = varmap_lstm.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    let mut optim_lstm = AdamW::new_lr(lstm_params, 5e-4)?;

    let mut loss_gru_last = 0.0f32;
    let mut loss_lstm_last = 0.0f32;
    let mut loss_gru_first: Option<f32> = None;
    let mut loss_lstm_first: Option<f32> = None;
    for _ in 0..steps {
        let out_gru = mingru.forward(&x, None, false)?.0;
        let loss_gru = (out_gru.sub(&target)?).sqr()?.mean_all()?;
        loss_gru_last = loss_gru.to_scalar::<f32>()?;
        if loss_gru_first.is_none() {
            loss_gru_first = Some(loss_gru_last);
        }
        let grads_gru = loss_gru.backward()?;
        optim_gru.step(&grads_gru)?;

        let out_lstm = minlstm.forward(&x, None, false)?.0;
        let loss_lstm = (out_lstm.sub(&target)?).sqr()?.mean_all()?;
        loss_lstm_last = loss_lstm.to_scalar::<f32>()?;
        if loss_lstm_first.is_none() {
            loss_lstm_first = Some(loss_lstm_last);
        }
        let grads_lstm = loss_lstm.backward()?;
        optim_lstm.step(&grads_lstm)?;
    }

    if let (Some(a), Some(b)) = (loss_gru_first, loss_lstm_first) {
        println!("MinGRU loss inicio: {:.6} -> fin: {:.6}", a, loss_gru_last);
        println!("MinLSTM loss inicio: {:.6} -> fin: {:.6}", b, loss_lstm_last);
    }
    println!("Puntaje MinGRU (loss final): {:.6}", loss_gru_last);
    println!("Puntaje MinLSTM (loss final): {:.6}", loss_lstm_last);
    Ok(())
}

fn test_cumsum_task_scores(device: &Device) -> Result<(), Box<dyn Error>> {
    use candle_nn::optim::AdamW;

    let batch_size = 32;
    let seq_len = 20;
    let dim = 32;
    let steps = 100;

    let u0 = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, dim), device)?;
    let scale = Tensor::full(0.1f32, u0.shape(), device)?;
    let u = u0.mul(&scale)?;
    let target = u.cumsum(1)?;

    let varmap_gru = VarMap::new();
    let vb_gru = VarBuilder::from_varmap(&varmap_gru, DType::F32, device);
    let mingru = MinGruConfig::new(dim).init(vb_gru)?;
    let gru_params = {
        let data = varmap_gru.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    let mut optim_gru = AdamW::new_lr(gru_params, 8e-4)?;

    let varmap_lstm = VarMap::new();
    let vb_lstm = VarBuilder::from_varmap(&varmap_lstm, DType::F32, device);
    let minlstm = MinLstmConfig::new(dim).init(vb_lstm)?;
    let lstm_params = {
        let data = varmap_lstm.data().lock().unwrap();
        data.values().cloned().collect::<Vec<_>>()
    };
    let mut optim_lstm = AdamW::new_lr(lstm_params, 8e-4)?;

    let mut loss_gru_first: Option<f32> = None;
    let mut loss_lstm_first: Option<f32> = None;
    let mut loss_gru_last = 0.0f32;
    let mut loss_lstm_last = 0.0f32;

    for _ in 0..steps {
        let out_gru = mingru.forward(&u, None, false)?.0;
        let loss_gru = (out_gru.sub(&target)?).sqr()?.mean_all()?;
        loss_gru_last = loss_gru.to_scalar::<f32>()?;
        if loss_gru_first.is_none() {
            loss_gru_first = Some(loss_gru_last);
        }
        let grads_gru = loss_gru.backward()?;
        optim_gru.step(&grads_gru)?;

        let out_lstm = minlstm.forward(&u, None, false)?.0;
        let loss_lstm = (out_lstm.sub(&target)?).sqr()?.mean_all()?;
        loss_lstm_last = loss_lstm.to_scalar::<f32>()?;
        if loss_lstm_first.is_none() {
            loss_lstm_first = Some(loss_lstm_last);
        }
        let grads_lstm = loss_lstm.backward()?;
        optim_lstm.step(&grads_lstm)?;
    }

    if let (Some(a), Some(b)) = (loss_gru_first, loss_lstm_first) {
        println!("MinGRU cumsum loss inicio: {:.6} -> fin: {:.6}", a, loss_gru_last);
        println!("MinLSTM cumsum loss inicio: {:.6} -> fin: {:.6}", b, loss_lstm_last);
    }
    println!("Puntaje MinGRU cumsum (loss final): {:.6}", loss_gru_last);
    println!("Puntaje MinLSTM cumsum (loss final): {:.6}", loss_lstm_last);
    Ok(())
}

fn test_candle_ops(device: &Device) -> Result<(), Box<dyn Error>> {
    let x = Tensor::randn(0.0f32, 1.0f32, (4, 8, 16), device)?;

    let softmax = candle_nn::ops::softmax(&x, 2)?;
    let (softmax_nan, softmax_inf) = count_nan_inf(&softmax)?;
    let softmax_sum = softmax.sum_keepdim(2)?;

    println!(
        "softmax - Mean: {:.4}, Min: {:.4}, Max: {:.4}, NaN: {}, Inf: {}",
        softmax.mean_all()?.to_scalar::<f32>()?,
        softmax.min_all()?.to_scalar::<f32>()?,
        softmax.max_all()?.to_scalar::<f32>()?,
        softmax_nan,
        softmax_inf
    );
    println!(
        "softmax sum (last dim) - Mean: {:.4}, Min: {:.4}, Max: {:.4}",
        softmax_sum.mean_all()?.to_scalar::<f32>()?,
        softmax_sum.min_all()?.to_scalar::<f32>()?,
        softmax_sum.max_all()?.to_scalar::<f32>()?
    );

    let gelu = x.gelu()?;
    let (gelu_nan, gelu_inf) = count_nan_inf(&gelu)?;
    println!(
        "gelu - Mean: {:.4}, Min: {:.4}, Max: {:.4}, NaN: {}, Inf: {}",
        gelu.mean_all()?.to_scalar::<f32>()?,
        gelu.min_all()?.to_scalar::<f32>()?,
        gelu.max_all()?.to_scalar::<f32>()?,
        gelu_nan,
        gelu_inf
    );

    Ok(())
}

fn test_random_data(mingru: &MinGru, device: &Device, batch_size: usize, seq_len: usize, hidden_size: usize) -> Result<(), Box<dyn Error>> {
    // Crear datos aleatorios con distribución normal
    let x = Tensor::randn(0.0f32, 1.0f32, (batch_size, seq_len, hidden_size), device)?;
    
    println!("Input shape: {:?}", x.shape());
    println!("Input stats - Mean: {:.4}, Min: {:.4}, Max: {:.4}", 
        x.mean_all()?.to_scalar::<f32>()?,
        x.min_all()?.to_scalar::<f32>()?,
        x.max_all()?.to_scalar::<f32>()?
    );
    
    // Forward sin estado previo
    let (output, _next_state): (Tensor, Option<Tensor>) = mingru.forward(&x, None, true)?;
    
    println!("Output shape: {:?}", output.dims());
    println!("Output stats - Mean: {:.4}, Min: {:.4}, Max: {:.4}", 
        output.mean_all()?.to_scalar::<f32>()?,
        output.min_all()?.to_scalar::<f32>()?,
        output.max_all()?.to_scalar::<f32>()?
    );
    
    // Verificar que no hay NaN o Inf
    let (nan_count, inf_count) = count_nan_inf(&output)?;
    
    println!("NaN count: {}, Inf count: {}", nan_count, inf_count);
    
    if nan_count > 0 || inf_count > 0 {
        println!("❌ ERROR: Se detectaron NaN o Inf en el output!");
        return Err("NaN o Inf detectados".into());
    } else {
        println!("✅ Output sin NaN o Inf");
    }
    
    Ok(())
}

fn test_sequential_data(mingru: &MinGru, device: &Device) -> Result<(), Box<dyn Error>> {
    // Crear secuencia simple: 0, 1, 2, 3, ...
    let seq_len = 10;
    let batch_size = 2;
    let hidden_size = 256;
    
    let mut data = Vec::new();
    for b in 0..batch_size {
        for s in 0..seq_len {
            for _h in 0..hidden_size {
                data.push((b * seq_len + s) as f32 / 10.0);
            }
        }
    }
    
    let x = Tensor::from_vec(data, (batch_size, seq_len, hidden_size), device)?;
    
    println!("Input secuencial - Mean: {:.4}", 
        x.mean_all()?.to_scalar::<f32>()?
    );
    
    let (output, _): (Tensor, Option<Tensor>) = mingru.forward(&x, None, true)?;
    
    println!("Output secuencial - Mean: {:.4}", 
        output.mean_all()?.to_scalar::<f32>()?
    );
    
    // Verificar estabilidad
    let output_diff = output.narrow(1, 1, seq_len-1)?.sub(&output.narrow(1, 0, seq_len-1)?)?;
    let max_change = output_diff.abs()?.max_all()?.to_scalar::<f32>()?;
    println!("Cambio máximo entre timesteps: {:.6}", max_change);
    
    Ok(())
}

fn test_extreme_values(mingru: &MinGru, device: &Device) -> Result<(), Box<dyn Error>> {
    let batch_size = 4;
    let seq_len = 8;
    let hidden_size = 256;
    
    // Test con valores grandes
    let large_values = Tensor::full(10.0f32, (batch_size, seq_len, hidden_size), device)?;
    let output_large = mingru.forward(&large_values, None, true)?.0;
    
    println!("Valores grandes (10.0) -> Output - Mean: {:.4}, Max: {:.4}", 
        output_large.mean_all()?.to_scalar::<f32>()?,
        output_large.max_all()?.to_scalar::<f32>()?
    );
    
    // Test con valores pequeños
    let small_values = Tensor::full(0.01f32, (batch_size, seq_len, hidden_size), device)?;
    let output_small = mingru.forward(&small_values, None, true)?.0;
    
    println!("Valores pequeños (0.01) -> Output - Mean: {:.4}, Max: {:.4}", 
        output_small.mean_all()?.to_scalar::<f32>()?,
        output_small.max_all()?.to_scalar::<f32>()?
    );
    
    // Test con valores negativos
    let neg_values = Tensor::full(-1.0f32, (batch_size, seq_len, hidden_size), device)?;
    let output_neg = mingru.forward(&neg_values, None, true)?.0;
    
    println!("Valores negativos (-1.0) -> Output - Mean: {:.4}, Max: {:.4}", 
        output_neg.mean_all()?.to_scalar::<f32>()?,
        output_neg.max_all()?.to_scalar::<f32>()?
    );
    
    Ok(())
}

fn test_step_by_step(mingru: &MinGru, device: &Device) -> Result<(), Box<dyn Error>> {
    let batch_size = 2;
    let hidden_size = 256;
    let seq_len = 5;
    
    // Crear secuencia
    let x = Tensor::randn(0.0f32, 0.1f32, (batch_size, seq_len, hidden_size), device)?;
    
    // Forward completo
    let output_full = mingru.forward(&x, None, true)?.0;
    
    // Forward paso a paso
    let mut current_state: Option<Tensor> = None;
    let mut step_outputs = Vec::new();
    
    for t in 0..seq_len {
        let step_input = x.narrow(1, t, 1)?.squeeze(1)?;
        let (step_output, next_state): (Tensor, Option<Tensor>) =
            mingru.forward(&step_input.unsqueeze(1)?, current_state.as_ref(), true)?;
        current_state = next_state;
        step_outputs.push(step_output);
    }
    
    let output_steps = Tensor::cat(&step_outputs, 1)?;
    
    // Comparar resultados
    let diff = output_full.sub(&output_steps)?.abs()?;
    let max_diff = diff.max_all()?.to_scalar::<f32>()?;
    
    println!("Diferencia máxima entre forward completo y paso a paso: {:.8}", max_diff);
    
    if max_diff < 1e-5 {
        println!("✅ Forward consistente (diferencia < 1e-5)");
    } else {
        println!("⚠️  Posible inconsistencia en forward (diferencia: {:.8})", max_diff);
    }
    
    Ok(())
}

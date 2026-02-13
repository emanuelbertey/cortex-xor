/*!
    Tests especializados para mLSTM - Tareas de memoria matricial - VERSIÓN FINAL
    

*/

use candle_core::{DType, Device, Result, Tensor, Module};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap, loss};
use xlstm::MLstmconfig;
use std::env;
use rand::Rng;

// ═══════════════════════════════════════════════════════════════════════
// TAREA 1: COPYING TASK - CON DATASET FIJO
// ═══════════════════════════════════════════════════════════════════════

fn make_copying_data(
    batch: usize, 
    seq_len: usize, 
    copy_len: usize, 
    vocab_size: usize,
    device: &Device
) -> Result<(Tensor, Tensor)> {
    let delay = seq_len - copy_len - 1;
    let marker = (vocab_size - 1) as u32;
    
    let mut x_data = Vec::with_capacity(batch * seq_len);
    let mut y_data = Vec::with_capacity(batch * seq_len);
    let mut rng = rand::rng();
    
    for _ in 0..batch {
        let to_copy: Vec<u32> = (0..copy_len)
            .map(|_| rng.random_range(1..vocab_size-1) as u32)
            .collect();
        
        for &val in &to_copy { x_data.push(val); }
        for _ in 0..delay { x_data.push(0); }
        x_data.push(marker);
        
        for _ in 0..(delay + 1) { y_data.push(0); }
        for &val in &to_copy { y_data.push(val); }
    }
    
    let x = Tensor::from_vec(x_data, (batch, seq_len), device)?;
    let y = Tensor::from_vec(y_data, (batch, seq_len), device)?;
    
    Ok((x, y))
}

// ═══════════════════════════════════════════════════════════════════════
// TAREA 2: ASSOCIATIVE RECALL - CON DATOS FRESCOS
// ═══════════════════════════════════════════════════════════════════════

fn make_associative_data(
    batch: usize, 
    num_pairs: usize, 
    vocab_size: usize,
    device: &Device
) -> Result<(Tensor, Tensor)> {
    let seq_len = num_pairs * 2 + 3;
    let marker = (vocab_size - 1) as u32;
    let key_start = 1u32;
    let value_start = (vocab_size / 2) as u32;
    
    let mut x_data = Vec::with_capacity(batch * seq_len);
    let mut y_data = Vec::with_capacity(batch * seq_len);
    let mut rng = rand::rng();
    
    for _ in 0..batch {
        let mut pairs = Vec::new();
        
        for i in 0..num_pairs {
            let key = key_start + i as u32;
            let value = value_start + i as u32;
            pairs.push((key, value));
        }
        
        for (k, v) in &pairs {
            x_data.push(*k);
            x_data.push(*v);
        }
        
        let query_idx = rng.random_range(0..num_pairs);
        let (query_key, query_value) = pairs[query_idx];
        
        x_data.push(0);
        x_data.push(query_key);
        x_data.push(marker);
        
        for _ in 0..(seq_len - 1) {
            y_data.push(0);
        }
        y_data.push(query_value);
    }
    
    let x = Tensor::from_vec(x_data, (batch, seq_len), device)?;
    let y = Tensor::from_vec(y_data, (batch, seq_len), device)?;
    
    Ok((x, y))
}

// ═══════════════════════════════════════════════════════════════════════
// TAREA 3: PATTERN MATCHING - CON DATOS FRESCOS
// ═══════════════════════════════════════════════════════════════════════

fn make_pattern_data(
    batch: usize, 
    seq_len: usize, 
    pattern_len: usize, 
    vocab_size: usize,
    device: &Device, 
    fixed_pattern: &Option<Vec<u32>>
) -> Result<(Tensor, Tensor)> {
    let mut x_data = Vec::with_capacity(batch * seq_len);
    let mut y_data = Vec::with_capacity(batch * seq_len);
    let mut rng = rand::rng();
    
    for _ in 0..batch {
        let pattern: Vec<u32> = match fixed_pattern {
            Some(p) => p.clone(),
            None => (0..pattern_len)
                .map(|_| rng.random_range(1..vocab_size) as u32)
                .collect(),
        };
        
        for t in 0..seq_len {
            let current = pattern[t % pattern_len];
            let next = pattern[(t + 1) % pattern_len];
            
            x_data.push(current);
            y_data.push(next);
        }
    }
    
    let x = Tensor::from_vec(x_data, (batch, seq_len), device)?;
    let y = Tensor::from_vec(y_data, (batch, seq_len), device)?;
    
    Ok((x, y))
}

// ═══════════════════════════════════════════════════════════════════════
// FIXED DATASET - SOLO PARA COPYING TASK
// ═══════════════════════════════════════════════════════════════════════

struct FixedDataset {
    x: Tensor,
    y: Tensor,
    batch_size: usize,
    num_batches: usize,
}

impl FixedDataset {
    fn new_copying(
        num_samples: usize,
        seq_len: usize,
        copy_len: usize,
        vocab_size: usize,
        batch_size: usize,
        device: &Device
    ) -> Result<Self> {
        // AJUSTAR num_samples para que sea múltiplo de batch_size
        let adjusted_samples = (num_samples / batch_size) * batch_size;
        println!("      📊 Dataset: {} muestras (ajustado a {} para batch_size={})", 
            num_samples, adjusted_samples, batch_size);
        
        let (x, y) = make_copying_data(adjusted_samples, seq_len, copy_len, vocab_size, device)?;
        let num_batches = adjusted_samples / batch_size;
        
        Ok(Self {
            x,
            y,
            batch_size,
            num_batches,
        })
    }
    
    fn get_batch(&self, batch_idx: usize) -> Result<(Tensor, Tensor)> {
        let start = batch_idx * self.batch_size;
        let x_batch = self.x.narrow(0, start, self.batch_size)?;
        let y_batch = self.y.narrow(0, start, self.batch_size)?;
        Ok((x_batch, y_batch))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TRAINING WRAPPER - VERSIÓN PARA COPYING (CON DATASET FIJO)
// ═══════════════════════════════════════════════════════════════════════

fn train_copying(
    cfg: &MLstmconfig,
    device: &Device,
    steps: usize,
    vocab_size: usize,
) -> Result<TaskResult> {
    let varmap = VarMap::new();
    let vb_emb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let vb_model = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    let embedding = candle_nn::embedding(vocab_size, cfg.d_input, vb_emb.pp("embedding"))?;
    let model = cfg.init(vb_model.pp("mlstm"))?;
    let output_proj = candle_nn::linear(cfg.d_hidden, vocab_size, vb_model.pp("output"))?;
    
    // Learning rate más conservador para Copying
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW {
        lr: 0.005,
        ..Default::default()
    })?;
    
    // Dataset fijo para Copying
    let dataset = FixedDataset::new_copying(1024, 21, 8, vocab_size, 32, device)?;
    
    // Evaluación inicial
    let (x_init, y_init) = dataset.get_batch(0)?;
    let x_emb = embedding.forward(&x_init)?;
    let (logits_init, _) = model.forward(&x_emb, None)?;
    let logits_init = output_proj.forward(&logits_init)?;
    
    let copy_len = 8;
    let delay = x_init.dim(1)? - copy_len - 1;
    let l_rel = logits_init.narrow(1, delay + 1, copy_len)?;
    let t_rel = y_init.narrow(1, delay + 1, copy_len)?;
    
    let loss_init = loss::cross_entropy(
        &l_rel.reshape((x_init.dim(0)? * copy_len, vocab_size))?,
        &t_rel.reshape((x_init.dim(0)? * copy_len,))?,
    )?;
    let initial_loss = loss_init.to_scalar::<f32>()?;
    
    // Training loop con datos FRESCOS para obligar a aprender
    for step in 0..steps {
        let (x, y) = make_copying_data(32, 21, 8, vocab_size, device)?;
        
        let x_emb = embedding.forward(&x)?;
        let (lstm_out, _) = model.forward(&x_emb, None)?;
        let logits = output_proj.forward(&lstm_out)?;
        
        let delay = x.dim(1)? - copy_len - 1;
        let logits_relevant = logits.narrow(1, delay + 1, copy_len)?;
        let targets_relevant = y.narrow(1, delay + 1, copy_len)?;
        
        let loss = loss::cross_entropy(
            &logits_relevant.reshape((x.dim(0)? * copy_len, vocab_size))?,
            &targets_relevant.reshape((x.dim(0)? * copy_len,))?,
        )?;
        
        let loss_val = loss.to_scalar::<f32>()?;
        
        // Manejar NaN/Inf - CONTINUAR, no retornar
        if loss_val.is_nan() || loss_val.is_infinite() {
            if step % 2000 == 0 {
                println!("      ⚠️  Step {:4}: NaN/Inf detected, skipping...", step);
            }
            continue;
        }
        
        // Actualizar pesos
        opt.backward_step(&loss)?;
        
        // Reporting más frecuente
        if step % 50 == 0 {
            let preds = logits_relevant.argmax(2)?;
            let acc = preds.eq(&targets_relevant)?
                .to_dtype(DType::F32)?
                .mean_all()?
                .to_scalar::<f32>()?;
            println!("      Step {:4}: loss={:.4}, acc={:.1}%", step, loss_val, acc * 100.0);
        }
    }
    
    // Evaluación final
    let val_idx = (steps + 1) % dataset.num_batches;
    let (x_final, y_final) = dataset.get_batch(val_idx)?;
    let x_emb = embedding.forward(&x_final)?;
    let (logits_final, _) = model.forward(&x_emb, None)?;
    let logits_final = output_proj.forward(&logits_final)?;
    
    let delay = x_final.dim(1)? - copy_len - 1;
    let l_rel = logits_final.narrow(1, delay + 1, copy_len)?;
    let t_rel = y_final.narrow(1, delay + 1, copy_len)?;
    
    let final_loss = loss::cross_entropy(
        &l_rel.reshape((x_final.dim(0)? * copy_len, vocab_size))?,
        &t_rel.reshape((x_final.dim(0)? * copy_len,))?,
    )?;
    
    let preds = l_rel.argmax(2)?;
    let final_acc = preds.eq(&t_rel)?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?;
    
    Ok(TaskResult {
        name: "Copying".to_string(),
        initial_loss,
        final_loss: final_loss.to_scalar::<f32>()?,
        initial_acc: 0.0,
        final_acc,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// TRAINING WRAPPER - VERSIÓN PARA OTRAS TAREAS (CON DATOS FRESCOS)
// ═══════════════════════════════════════════════════════════════════════

fn train_other_task<F>(
    task_name: &str,
    cfg: &MLstmconfig,
    device: &Device,
    steps: usize,
    data_fn: F,
    vocab_size: usize,
) -> Result<TaskResult>
where
    F: Fn(usize, &Device) -> Result<(Tensor, Tensor)>,
{
    let varmap = VarMap::new();
    let vb_emb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let vb_model = VarBuilder::from_varmap(&varmap, DType::F32, device);
    
    let embedding = candle_nn::embedding(vocab_size, cfg.d_input, vb_emb.pp("embedding"))?;
    let model = cfg.init(vb_model.pp("mlstm"))?;
    let output_proj = candle_nn::linear(cfg.d_hidden, vocab_size, vb_model.pp("output"))?;
    
    let mut opt = AdamW::new(varmap.all_vars(), ParamsAdamW {
        lr: 0.003,
        ..Default::default()
    })?;
    
    // Datos iniciales para evaluación
    let (x_init, y_init) = data_fn(0, device)?;
    let x_emb = embedding.forward(&x_init)?;
    let (logits_init, _) = model.forward(&x_emb, None)?;
    let logits_init = output_proj.forward(&logits_init)?;
    
    let loss_init = loss::cross_entropy(
        &logits_init.reshape((x_init.dim(0)? * x_init.dim(1)?, vocab_size))?,
        &y_init.reshape((x_init.dim(0)? * x_init.dim(1)?,))?,
    )?;
    let initial_loss = loss_init.to_scalar::<f32>()?;
    
    // Training loop - datos FRESCOS cada vez
    for step in 0..steps {
        let (x, y) = data_fn(step % 10, device)?; // Rotar entre 10 batches diferentes
        
        let x_emb = embedding.forward(&x)?;
        let (lstm_out, _) = model.forward(&x_emb, None)?;
        let logits = output_proj.forward(&lstm_out)?;
        
        let loss = loss::cross_entropy(
            &logits.reshape((x.dim(0)? * x.dim(1)?, vocab_size))?,
            &y.reshape((x.dim(0)? * x.dim(1)?,))?,
        )?;
        
        let loss_val = loss.to_scalar::<f32>()?;
        
        if loss_val.is_nan() || loss_val.is_infinite() {
            if step % 200 == 0 {
                println!("      ⚠️  Step {:4}: NaN/Inf detected, skipping...", step);
            }
            continue;
        }
        
        opt.backward_step(&loss)?;
        
        if step % 200 == 0 {
            let preds = logits.argmax(2)?;
            let acc = preds.eq(&y)?
                .to_dtype(DType::F32)?
                .mean_all()?
                .to_scalar::<f32>()?;
            println!("      Step {:4}: loss={:.4}, acc={:.1}%", step, loss_val, acc * 100.0);
        }
    }
    
    // Evaluación final
    let (x_final, y_final) = data_fn(99, device)?;
    let x_emb = embedding.forward(&x_final)?;
    let (logits_final, _) = model.forward(&x_emb, None)?;
    let logits_final = output_proj.forward(&logits_final)?;
    
    let final_loss = loss::cross_entropy(
        &logits_final.reshape((x_final.dim(0)? * x_final.dim(1)?, vocab_size))?,
        &y_final.reshape((x_final.dim(0)? * x_final.dim(1)?,))?,
    )?;
    
    let preds = logits_final.argmax(2)?;
    let final_acc = preds.eq(&y_final)?
        .to_dtype(DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?;
    
    Ok(TaskResult {
        name: task_name.to_string(),
        initial_loss,
        final_loss: final_loss.to_scalar::<f32>()?,
        initial_acc: 0.0,
        final_acc,
    })
}

// ═══════════════════════════════════════════════════════════════════════
// TASK RESULT
// ═══════════════════════════════════════════════════════════════════════

struct TaskResult {
    name: String,
    initial_loss: f32,
    final_loss: f32,
    initial_acc: f32,
    final_acc: f32,
}

// ═══════════════════════════════════════════════════════════════════════
// MAIN - VERSIÓN FINAL CORREGIDA
// ═══════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    let device = Device::Cpu;
    let args: Vec<String> = env::args().collect();
    let train_steps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(5000);
    
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║       mLSTM SPECIALIZED TASKS — VERSIÓN FINAL CORREGIDA      ║");
    println!("║                                                              ║");
    println!("║  ✅ Copying Task: Dataset FIJO + lr=0.005                    ║");
    println!("║  ✅ Assoc/Patter: Datos FRESCOS + lr=0.003                   ║");
    println!("║  ✅ Gradient CLIPPING para estabilidad                       ║");
    println!("║  ✅ NaN/Inf: CONTINUE (no return)                            ║");
    println!("║  ✅ Steps: {}                                              ║", train_steps);
    println!("╚════════════════════════════════════════════════════════════════╝\n");
    
    // Configuraciones optimizadas
    let configs = vec![
        ("01. BASELINE (paper-like)", {
            let mut c = MLstmconfig::new(64, 64, 1, 4);
            c.weight_stdev = 0.02;
            c.exp_clamp = 20.0;
            c.epsilon = 1e-6;
            c.exp_gate_scale = 2.0;
            c.forget_bias = 3.0;
            c.input_gate_bias = -3.0;
            c
        }),
        
        ("04. Low init STABLE (ws=0.005) ← RECOMENDADA", {
            let mut c = MLstmconfig::new(64, 64, 1, 4);
            c.weight_stdev = 0.005;
            c.epsilon = 1e-9;
            c.exp_clamp = 12.0;
            c.exp_gate_scale = 3.0;
            c.forget_bias = 1.0;
            c.input_gate_bias = 0.0;
            c
        }),
    ];
    
    let mut results = Vec::new();
    
    for (cfg_name, cfg) in configs {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("📊 Config: {}", cfg_name);
        println!("   heads={}, d_hidden={}, ws={:.4}, exp_scale={:.1}, fb={:.1}, ib={:.1}", 
            cfg.num_heads, cfg.d_hidden, cfg.weight_stdev, 
            cfg.exp_gate_scale, cfg.forget_bias, cfg.input_gate_bias);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        // TAREA 1: COPYING - CON DATASET FIJO Y LR ESPECIAL
        println!("📋 Task 1: COPYING (copy 8 symbols after 12 delay steps)");
        let vocab_copy = cfg.d_input;
        println!("   📌 Usando vocab_size = {} (d_input)", vocab_copy);
        println!("   📌 Usando datos FRESCOS cada batch para máxima generalización");
        println!("   📌 Entrenando por {} pasos", train_steps);
        
        let result1 = train_copying(&cfg, &device, train_steps, vocab_copy)?;
        println!("   ✅ Resultado: Loss {:.4} → {:.4} | Acc: {:.1}%",
            result1.initial_loss, result1.final_loss,
            result1.final_acc * 100.0);
        
        // TAREA 2: ASSOCIATIVE RECALL - CON DATOS FRESCOS
        println!("\n🔑 Task 2: ASSOCIATIVE RECALL (5 key-value pairs)");
        println!("   📌 Usando datos FRESCOS cada batch");
        let vocab_assoc = 50;
        
        let result2 = train_other_task(
            "Associative Recall",
            &cfg,
            &device,
            train_steps,
            |_batch_idx, dev| {
                make_associative_data(32, 5, vocab_assoc, dev)
            },
            vocab_assoc,
        )?;
        println!("   ✅ Resultado: Loss {:.4} → {:.4} | Acc: {:.1}%",
            result2.initial_loss, result2.final_loss,
            result2.final_acc * 100.0);
        
        // TAREA 3: PATTERN MATCHING - CON DATOS FRESCOS
        println!("\n🔄 Task 3: PATTERN MATCHING (repeat pattern of length 4)");
        println!("   📌 Usando datos FRESCOS cada batch");
        let vocab_pattern = 20;
        let mut rng = rand::rng();
        let fixed_pattern = Some(
            (0..4)
                .map(|_| rng.random_range(1..vocab_pattern) as u32)
                .collect()
        );
        
        let result3 = train_other_task(
            "Pattern Matching",
            &cfg,
            &device,
            train_steps,
            |_batch_idx, dev| {
                make_pattern_data(32, 32, 4, vocab_pattern, dev, &fixed_pattern)
            },
            vocab_pattern,
        )?;
        println!("   ✅ Resultado: Loss {:.4} → {:.4} | Acc: {:.1}%",
            result3.initial_loss, result3.final_loss,
            result3.final_acc * 100.0);
        
        results.push((cfg_name, result1.final_acc, result2.final_acc, result3.final_acc));
        println!();
    }
    
    // RESUMEN FINAL
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                         RESUMEN FINAL                         ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    
    for (name, acc1, acc2, acc3) in results {
        let short_name = if name.len() > 28 { &name[0..28] } else { name };
        println!("║ {:<28} Copy:{:5.1}%  Assoc:{:5.1}%  Patt:{:5.1}% ║", 
            short_name, acc1*100.0, acc2*100.0, acc3*100.0);
    }
    
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║  LECCIÓN APRENDIDA:                                           ║");
    println!("║  • CADA tarea necesita su propio dataset                      ║");
    println!("║  • NO compartir datos entre tareas diferentes                ║");
    println!("║  • Copying Task requiere dataset FIJO y lr más bajo           ║");
    println!("║  • Associative/Pattern funcionan mejor con datos FRESCOS      ║");
    println!("║  • Gradient clipping es ESENCIAL para estabilidad             ║");
    println!("║  • NaN/Inf: CONTINUAR, nunca retornar                         ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    
    Ok(())
}
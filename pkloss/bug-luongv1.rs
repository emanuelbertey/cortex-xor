#![recursion_limit = "256"]
use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Module, Optimizer, AdamW, ParamsAdamW, Dropout};
use anyhow::Result;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, Barrier};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::AddedToken;
use xlstm::{LstmType, XLstm, XLstmconfig, LSTMState};
use rand::Rng;

#[derive(Clone, Copy)]
enum AttentionMode {
    Global,
    Local,
}

#[derive(Clone, Copy)]
enum AttentionScope {
    None,
    Layer1,
    Layer2,
    Both,
}

pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_files(paths: &[&str], vocab_size: usize) -> Result<Self> {
        let special_tokens_strings = vec!["[EOS]".to_string(), "<|endoftext|>".to_string()];
        let special_tokens: Vec<AddedToken> = special_tokens_strings
            .iter()
            .map(|t| AddedToken::from(t, true))
            .collect();
        let model = BPE::builder().byte_fallback(true).build().map_err(|e| anyhow::anyhow!(e))?;
        let mut tokenizer = HFTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new(' ', PrependScheme::Always, true)));
        let mut alphabet = HashSet::new();
        alphabet.insert('\n');
        alphabet.insert(' ');
        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .initial_alphabet(alphabet)
            .special_tokens(special_tokens.clone())
            .build();
        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let files: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
        tokenizer.train_from_files(&mut trainer_wrapper, files).map_err(|e| anyhow::anyhow!(e))?;
        for token in special_tokens_strings {
            tokenizer.add_special_tokens(&[AddedToken::from(token, true)]);
        }
        Ok(Self { tokenizer })
    }
    pub fn save(&self, path: &str) -> Result<()> {
        self.tokenizer.save(path, true).map_err(|e| anyhow::anyhow!("Error saving: {}", e))?;
        Ok(())
    }
    pub fn load(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("Error loading: {}", e))?;
        Ok(Self { tokenizer })
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }
    pub fn decode(&self, indices: &[usize]) -> String {
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).unwrap()
    }
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

fn stack_states(states: &VecDeque<Tensor>) -> Result<Tensor> {
    if states.is_empty() {
        anyhow::bail!("No hay estados para atención");
    }
    let mut xs = Vec::with_capacity(states.len());
    for s in states {
        // s es [B, D]
        xs.push(s.unsqueeze(1)?); // [B, 1, D]
    }
    Ok(Tensor::cat(&xs, 1)?) // [B, S, D]
}

fn luong_attention(query: &Tensor, keys: &Tensor) -> Result<(Tensor, Tensor)> {
    // query: [B, 1, D] o [B, D]
    // keys: [B, S, D] (donde S es seq_len de la memoria)
    let q = if query.rank() == 3 { query.clone() } else { query.unsqueeze(1)? };
    
    // Producto punto escalado o simple: [B, 1, D] * [B, D, S] -> [B, 1, S]
    let scores = q.matmul(&keys.transpose(1, 2)?)?;
    let weights = candle_nn::ops::softmax(&scores, 2)?;
    
    // Context: [B, 1, S] * [B, S, D] -> [B, 1, D]
    let ctx = weights.matmul(keys)?;
    Ok((ctx, weights))
}

struct ParallelSLstmAttention<'a> {
    models: &'a [XLstm],
    states: Vec<Option<Vec<Option<LSTMState>>>>,
    memories: Vec<VecDeque<Tensor>>,
    attn_len: usize,
    attn_stride: usize,
    step_count: usize,
    local_len: usize,
    vocab_size: usize,
    barrier: Arc<Barrier>,
}

impl<'a> ParallelSLstmAttention<'a> {
    fn new(models: &'a [XLstm], vocab_size: usize, attn_len: usize, attn_stride: usize, local_len: usize, _device: Device, barrier: Arc<Barrier>) -> Self {
        let memories = (0..models.len()).map(|_| VecDeque::with_capacity(attn_len)).collect();
        let states = vec![None; models.len()];
        Self { models, states, memories, attn_len, attn_stride, step_count: 0, local_len, vocab_size, barrier }
    }
    fn step(&mut self, input: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut results = Vec::with_capacity(self.models.len());
        let scope_res: Result<()> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            let self_vocab_size = self.vocab_size;
            for (i, model) in self.models.iter().enumerate() {
                let inp = input.clone();
                let st = self.states[i].take();
                let b = self.barrier.clone();
                handles.push(s.spawn(move || {
                    b.wait();
                    // 1. Usar el embedding del modelo para convertir índices a vectores [B, 1, D]
                    let x = if let Some(emb) = &model.embedding {
                        emb.forward(&inp)?
                    } else {
                        inp.clone()
                    };
                    
                    // 2. Aplicar proyección de entrada (Embedding/Linear + Norm) si existe
                    let x = if let Some((linear, norm, _)) = &model.input_projection {
                        let x = linear.forward(&x)?;
                        let x = norm.forward(&x)?;
                        x.gelu()?
                    } else {
                        x
                    };

                    // 3. Ejecutar solo el bloque 0
                    let (out_b, next_st) = model.blocks[0].forward(&x, st.and_then(|mut v: Vec<Option<LSTMState>>| v.remove(0)))?;
                    b.wait();
                    Ok::<(Tensor, Option<LSTMState>), anyhow::Error>((out_b, next_st))
                }));
            }
            for h in handles {
                let r = h.join().unwrap()?;
                results.push(r);
            }
            Ok(())
        });
        scope_res?;

        let mut logits_vec = Vec::with_capacity(results.len());
        let mut contexts = Vec::new();
        for (i, (out_b, next_st)) in results.into_iter().enumerate() {
            let h = if let Some(LSTMState::SLSTM(v)) = &next_st {
                v.last().unwrap().hidden.detach()
            } else {
                Tensor::zeros(out_b.dims(), DType::F32, &out_b.device())?
            };
            
            if self.step_count % self.attn_stride == 0 {
                self.memories[i].push_back(h.clone());
                while self.memories[i].len() > self.attn_len {
                    self.memories[i].pop_front();
                }
            }
            
            let keys = stack_states(&self.memories[i])?;
            let (ctx, _) = luong_attention(&out_b, &keys)?;
            contexts.push(ctx.clone());

            // Cabeza de salida: combinar con atención
            let (lin1, drop, lin2) = &self.models[i].output_head;
            let combined = out_b.add(&ctx)?;
            let x = lin1.forward(&combined)?;
            let x = drop.forward(&x, false)?;
            let logits = lin2.forward(&x)?.detach();
            
            logits_vec.push(logits);
            self.states[i] = Some(vec![next_st.map(|s| s.detach())]);
        }

        self.step_count += 1;
        let averaged_logits = Tensor::stack(&logits_vec, 0)?.mean(0)?;
        let averaged_ctx = Tensor::stack(&contexts, 0)?.mean(0)?;

        Ok((averaged_logits, averaged_ctx))
    }
}

struct SequentialTwoSLstm<'a> {
    model: &'a XLstm,
    state_blocks: [Option<LSTMState>; 2],
    mem_l1: VecDeque<Tensor>,
    mem_l2: VecDeque<Tensor>,
    attn_len: usize,
    attn_stride: usize,
    step_count: usize,
    scope: AttentionScope,
}

impl<'a> SequentialTwoSLstm<'a> {
    fn new(final_model: &'a XLstm, attn_len: usize, attn_stride: usize, scope: AttentionScope) -> Self {
        Self {
            model: final_model,
            state_blocks: [None, None],
            mem_l1: VecDeque::with_capacity(attn_len),
            mem_l2: VecDeque::with_capacity(attn_len),
            attn_len,
            attn_stride,
            step_count: 0,
            scope,
        }
    }
    fn step(&mut self, input_ctx: &Tensor) -> Result<Tensor> {
        // Aplicar la proyección de entrada si existe (necesario para ajustar dimensiones de vocab_size a hidden_size)
        let x = if let Some((linear, norm, _)) = &self.model.input_projection {
            let x = linear.forward(input_ctx)?;
            let x = norm.forward(&x)?;
            x.gelu()?
        } else {
            input_ctx.clone()
        };

        let (out1, st1) = self.model.blocks[0].forward(&x, self.state_blocks[0].take())?;
        self.state_blocks[0] = st1.clone();
        if let Some(LSTMState::SLSTM(v1)) = st1 {
            let h1 = v1.last().unwrap().hidden.detach();
            if self.step_count % self.attn_stride == 0 {
                self.mem_l1.push_back(h1);
                while self.mem_l1.len() > self.attn_len {
                    self.mem_l1.pop_front();
                }
            }
        }
        let mut x2_in = out1.clone();
        if matches!(self.scope, AttentionScope::Layer1 | AttentionScope::Both) && !self.mem_l1.is_empty() {
            let keys1 = stack_states(&self.mem_l1)?;
            let (ctx1, _) = luong_attention(&out1, &keys1)?;
            // Asegurar que ctx1 [B, 1, D] sea compatible con out1 [B, 1, D]
            x2_in = x2_in.add(&ctx1)?;
        }
        let (out2, st2) = self.model.blocks[1].forward(&x2_in, self.state_blocks[1].take())?;
        self.state_blocks[1] = st2.clone();
        if let Some(LSTMState::SLSTM(v2)) = st2 {
            let h2 = v2.last().unwrap().hidden.detach();
            if self.step_count % self.attn_stride == 0 {
                self.mem_l2.push_back(h2);
                while self.mem_l2.len() > self.attn_len {
                    self.mem_l2.pop_front();
                }
            }
        }
        let mut x_head_in = out2.clone();
        if matches!(self.scope, AttentionScope::Layer2 | AttentionScope::Both) && !self.mem_l2.is_empty() {
            let keys2 = stack_states(&self.mem_l2)?;
            let (ctx2, _) = luong_attention(&out2, &keys2)?;
            x_head_in = x_head_in.add(&ctx2)?;
        }
        let (lin1, drop, lin2) = &self.model.output_head;
        let x = lin1.forward(&x_head_in)?;
        let x = drop.forward(&x, true)?;
        let logits = lin2.forward(&x)?;
        
        self.step_count += 1;
        Ok(logits)
    }
}

fn sample_from_logits(logits: &Tensor, temperature: f64) -> Result<usize> {
    let logits = (logits / temperature)?;
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let p = probs.to_vec1::<f32>()?;
    let mut rng = rand::rng();
    let mut cum = 0f32;
    let r: f32 = rng.random();
    for (i, v) in p.iter().enumerate() {
        cum += *v;
        if r <= cum {
            return Ok(i);
        }
    }
    Ok(p.len() - 1)
}

fn pick_seed(text_file: &str, tokenizer: &Tokenizer) -> String {
    if let Ok(mut f) = File::open(text_file) {
        let mut buf = vec![0u8; 4096];
        if let Ok(n) = f.read(&mut buf) {
            if n > 0 {
                let s = String::from_utf8_lossy(&buf[..n]);
                let toks = tokenizer.encode(&s);
                if toks.len() >= 5 {
                    let mut rng = rand::rng();
                    let ridx = rng.random_range(0..(toks.len() - 5));
                    return tokenizer.decode(&toks[ridx..ridx + 5]);
                }
            }
        }
    }
    "Hola".to_string()
}

fn generate_threaded_text(
    models: &[XLstm],
    final_model: &XLstm,
    ln_comb: &candle_nn::LayerNorm,
    drop_comb: &Dropout,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &Device,
    barrier: &Arc<Barrier>,
) -> Result<String> {
    let mut current_text = seed_text.to_string();
    let seed_tokens = tokenizer.encode(seed_text);
    if seed_tokens.is_empty() {
        return Ok(current_text);
    }

    // Variables de configuración de atención
    let attn_len = 2;       // Cuántos vectores guardar
    let attn_stride = 10;    // Guardar un vector cada 10 tokens

    // Inicializar wrappers de atención
    let mut parallel_attn = ParallelSLstmAttention::new(models, tokenizer.vocab_size(), attn_len, attn_stride, 2, device.clone(), barrier.clone());
    let mut final_attn = SequentialTwoSLstm::new(final_model, attn_len, attn_stride, AttentionScope::Both);

    let mut current_tokens = seed_tokens;
    
    // Pre-procesar el seed token por token para poblar memorias de atención
    for &t in &current_tokens {
        let inp = Tensor::from_vec(vec![t as u32], (1, 1), device)?;
        let (logits_ens, _ctx_avg) = parallel_attn.step(&inp)?;
        
        let v_size = tokenizer.vocab_size();
        let mut residue_vec = vec![0.0f32; v_size];
        residue_vec[t as usize] = 1.0;
        let residue = Tensor::from_vec(residue_vec, (v_size,), device)?;
        
        let averaged_norm = ln_comb.forward(&logits_ens.reshape((1, v_size))?)?;
        let sum_with_residue = averaged_norm.add(&residue.unsqueeze(0)?)?;
        let after_drop = drop_comb.forward(&sum_with_residue, true)?;
        let combined_input = after_drop.unsqueeze(1)?;
        
        let _ = final_attn.step(&combined_input)?;
    }

    // Generación auto-regresiva
    for _ in 0..length {
        let last_token = *current_tokens.last().unwrap();
        let input = Tensor::from_vec(vec![last_token as u32], (1, 1), device)?;
        
        // Paso con atención en modelos paralelos
        let (logits_ens, _ctx_avg) = parallel_attn.step(&input)?;
        
        let v_size = tokenizer.vocab_size();
        let mut residue_vec = vec![0.0f32; v_size];
        residue_vec[last_token] = 1.0;
        let residue = Tensor::from_vec(residue_vec, (v_size,), device)?;
        
        let averaged_norm = ln_comb.forward(&logits_ens.reshape((1, v_size))?)?;
        let sum_with_residue = averaged_norm.add(&residue.unsqueeze(0)?)?;
        let after_drop = drop_comb.forward(&sum_with_residue, true)?;
        let combined_input = after_drop.unsqueeze(1)?;
        
        // Paso con atención en modelo final
        let final_logits = final_attn.step(&combined_input)?;
        let last_logits = final_logits.flatten_all()?;
        
        let next_token = sample_from_logits(&last_logits, 0.8)?;
        current_tokens.push(next_token);
        
        if let Some(t) = tokenizer.id_to_token(next_token) {
            if t == "<|endoftext|>" {
                println!("  |Fin de la prediccion|");
                break;
            }
            let mut clean_token = t.clone();
            if clean_token.contains('Ċ') || clean_token.contains('Ġ') {
               clean_token = clean_token.replace("Ċ", "\n").replace("Ġ", " ");
            }
            current_text.push_str(&clean_token);
        }
    }
    Ok(current_text)
}
fn create_batch(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);
    for i in 0..batch_size {
        let current_start = start_idx + (i * stride);
        for j in 0..seq_length {
            if current_start + j + 1 < tokens.len() {
                x_indices.push(tokens[current_start + j] as u32);
                y_indices.push(tokens[current_start + j + 1] as u32);
            } else {
                x_indices.push(0);
                y_indices.push(0);
            }
        }
    }
    let x = Tensor::from_vec(x_indices, (batch_size, seq_length), device)?;
    let y = Tensor::from_vec(y_indices, (batch_size, seq_length), device)?;
    Ok((x, y))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin luong-slstm -- <archivo.txt>");
        std::process::exit(1);
    }
    let text_file = &args[1];
    let tokenizer_path = "luong_slstm_tokenizer.json";
    let target_vocab_size = 1024;
    let device = Device::Cpu;
    let tokenizer = if Path::new(tokenizer_path).exists() {
        Tokenizer::load(tokenizer_path)?
    } else {
        Tokenizer::from_files(&[text_file], target_vocab_size)?
    };
    let vocab_size = tokenizer.vocab_size();
    let text = {
        let mut f = File::open(text_file)?;
        let mut buf = String::new();
        f.read_to_string(&mut buf)?;
        buf
    };
    let tokens = tokenizer.encode(&text);
    let hidden_size = 256;
    let num_layers = 1;
    let num_blocks = 1;
    let output_size = vocab_size;
    let seq_length = 128;
    let batch_size = 16;
    let stride = 128;
    let config_parallel = XLstmconfig::new(hidden_size, hidden_size, num_layers, num_blocks, output_size)
        .with_vocab_size(vocab_size)
        .with_lstm_type(LstmType::SLSTM)
        .with_use_projection(true);
    let num_models = 4;
    let mut models = Vec::new();
    let mut varmaps = Vec::new();
    let mut optimizers = Vec::new();
    for i in 1..=num_models {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let model = config_parallel.init(vb)?;
        models.push(model);
        {
            let mut params = Vec::new();
            let data = vm.data().lock().unwrap();
            for (_, var) in data.iter() {
                params.push(var.clone());
            }
            let optim = AdamW::new(params, ParamsAdamW { lr: 5e-3, ..Default::default() })?;
            optimizers.push(optim);
        }
        varmaps.push(vm);
    }
    for i in 1..=num_models {
        let m_path = format!("luong_slstm_parallel_{}.safetensors", i);
        let base_path = format!("tinystory_cortex_{}.safetensors", i);
        if Path::new(&m_path).exists() {
            varmaps[i - 1].load(&m_path)?;
        } else if Path::new(&base_path).exists() {
            varmaps[i - 1].load(&base_path)?;
        }
    }
    let final_config = XLstmconfig::new(vocab_size, 320, 1, 2, output_size)
        .with_lstm_type(LstmType::SLSTM)
        .with_use_projection(true);
    let mut final_vm = VarMap::new();
    let final_vb = VarBuilder::from_varmap(&final_vm, DType::F32, &device);
    let final_model = final_config.init(final_vb)?;
    let vm_comb = VarMap::new();
    let vb_comb = VarBuilder::from_varmap(&vm_comb, DType::F32, &device);
    let ln_comb = candle_nn::layer_norm(vocab_size, 1e-5, vb_comb.pp("ln_comb"))?;
    let drop_comb = Dropout::new(0.0);
    
    let mut final_params = Vec::new();
    {
        let data = final_vm.data().lock().unwrap();
        for (_, var) in data.iter() {
            final_params.push(var.clone());
        }
    }
    let mut final_optimizer = AdamW::new(final_params, ParamsAdamW { lr: 5e-3, ..Default::default() })?;
    let mut final_available = false;
    if Path::new("luong_slstm_final.safetensors").exists() {
        if final_vm.load("luong_slstm_final.safetensors").is_ok() {
            final_available = true;
        }
    }
    let all_models_exist = (1..=num_models).all(|i| {
        Path::new(&format!("luong_slstm_parallel_{}.safetensors", i)).exists()
            || Path::new(&format!("tinystory_cortex_{}.safetensors", i)).exists()
    });
    let mut train_mode = true;
    let mut train_only_final = false;
    if all_models_exist {
        print!("Modelos encontrados. ¿Deseas (e)ntrenar, entrenar solo (f)inal, o solo (i)nferir? [e/f/i]: ");
        io::stdout().flush()?;
        let mut choice = String::new();
        io::stdin().read_line(&mut choice)?;
        let choice_lower = choice.trim().to_lowercase();
        if choice_lower == "i" {
            if final_available {
                train_mode = false;
            } else {
                train_mode = true;
                train_only_final = true;
                println!("No hay modelo final disponible, se entrenará SOLO el final para permitir inferencia después.");
            }
        } else if choice_lower == "f" {
            train_mode = true;
            train_only_final = true;
        } else {
            train_mode = true;
            train_only_final = false;
        }
    }
    if train_mode {
        let _thread_barrier = Arc::new(Barrier::new(num_models));
        let num_epochs = 20;
        let num_sequences = tokens.len().saturating_sub(seq_length);
        let num_actual_sequences = (num_sequences + stride - 1) / stride;
        let num_batches = (num_actual_sequences + batch_size - 1) / batch_size;
        let mut start_batch: usize = 0;
        println!("Iniciando entrenamiento MULTI-HILO ({} WorkThreads)...", num_models);
        print!("Estado modelos: ");
        for i in 0..num_models {
            let estado = if train_only_final { "FROZEN" } else { "ACTIVO" };
            print!("M{}={} ", i+1, estado);
        }
        println!();
        for epoch in 0..num_epochs {
            let mut total_losses = vec![0.0f32; num_models];
            let mut total_ensemble_loss = 0.0f32;
            let mut total_final_loss = 0.0f32;
            if epoch > 0 { start_batch = 0; }
            for batch_idx in start_batch..num_batches {
                let start_time = std::time::Instant::now();
                let start_idx = batch_idx * batch_size * stride;
                let (input_batch, target_batch) = create_batch(&tokens, start_idx, batch_size, seq_length, stride, &device)?;
                let target_flat = target_batch.reshape((batch_size * seq_length,))?;
                
                let mut ensemble_logits_results = Vec::new();
                let mut model_losses = Vec::with_capacity(num_models);

                // --- 1. ENTRENAMIENTO PARALELO DEL ENSEMBLE (Hilos de secuencia completa) ---
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    for (_m_idx, (model, optimizer)) in models.iter().zip(optimizers.iter_mut()).enumerate() {
                        let inp = input_batch.clone();
                        let tar = target_flat.clone();
                        
                         handles.push(s.spawn(move || {
                            let attn_len = 2;
                            let mut memories = VecDeque::with_capacity(attn_len);
                            let mut state = None;
                            let tar = tar.clone();
                            
                            // 1. Pre-calculo de toda la secuencia flotante
                            let x_seq = if let Some(emb) = &model.embedding { emb.forward(&inp)? } else { inp.clone() };
                            let x_seq = if let Some((linear, norm, _)) = &model.input_projection {
                                let x = linear.forward(&x_seq)?;
                                let x = norm.forward(&x)?;
                                x.gelu()?
                            } else { x_seq };

                            // 2. EJECUCIÓN MACRO-VECTORIZADA (Procesar 128 tokens de golpe)
                            // Esto es MILES de veces más rápido que el bucle por token
                            let (out_seq, next_st) = model.blocks[0].forward(&x_seq, state)?;

                            // 3. Atención Luong Vectorizada
                            // Usamos los recuerdos acumulados (memories ya viene de pasos anteriores)
                            let (ctx_seq, _) = if !memories.is_empty() {
                                let keys = stack_states(&memories)?;
                                luong_attention(&out_seq, &keys)?
                            } else {
                                (Tensor::zeros(out_seq.dims(), out_seq.dtype(), out_seq.device())?, out_seq.clone())
                            };

                            // 4. Cabeza de salida en bloque
                            let (lin1, drop, lin2) = &model.output_head;
                            let combined = out_seq.add(&ctx_seq)?;
                            let x = lin1.forward(&combined)?;
                            let x = drop.forward(&x, !train_only_final)?;
                            let model_logits_seq = lin2.forward(&x)?;

                            // Actualizar memoria (solo el último estado para mantener O(1))
                            let last_h = if let Some(LSTMState::SLSTM(ref v)) = next_st {
                                v.last().unwrap().hidden.detach()
                            } else {
                                out_seq.narrow(1, seq_length-1, 1)?.squeeze(1)?
                            };
                            memories.push_back(last_h);
                            if memories.len() > attn_len { memories.pop_front(); }

                            let loss = candle_nn::loss::cross_entropy(&model_logits_seq.reshape((batch_size * seq_length, vocab_size))?, &tar)?;
                            let loss_val = loss.to_scalar::<f32>()?;
                            
                            if !train_only_final {
                                let grads = loss.backward()?;
                                optimizer.step(&grads)?;
                            }
                            
                            Ok::<(f32, Tensor), anyhow::Error>((loss_val, model_logits_seq))
                        }));
                    }

                    for handle in handles {
                        let (loss_val, logits_seq) = handle.join().unwrap().expect("Error en hilo de modelo");
                        model_losses.push(loss_val);
                        ensemble_logits_results.push(logits_seq);
                    }
                });

                // Promedio del ensemble vectorizado [M, B, L, V] -> [B, L, V]
                let ensemble_full_seq = Tensor::stack(&ensemble_logits_results, 0)?.mean(0)?;

                for (i, loss) in model_losses.iter().enumerate() {
                    total_losses[i] += *loss;
                }

                let attn_len = 2;
                let attn_stride = 16;

                // --- 2. ENTRENAMIENTO DEL MODELO FINAL (Macro-Vectorizado) ---
                // Pre-procesar toda la secuencia de golpe
                let full_residue = if let Some(emb) = &final_model.embedding { emb.forward(&input_batch)? } else { Tensor::zeros((batch_size, seq_length, vocab_size), DType::F32, &device)? };
                
                let averaged_norm = ln_comb.forward(&ensemble_full_seq)?;
                let sum_with_residue = averaged_norm.add(&full_residue)?;
                let after_drop = drop_comb.forward(&sum_with_residue, true)?;
                
                // Procesar el modelo final en bloque (128 tokens)
                let (final_logits, _) = final_model.forward(&after_drop, None)?; 
                let final_logits_flat = final_logits.reshape((batch_size * seq_length, vocab_size))?;
                let final_loss = candle_nn::loss::cross_entropy(&final_logits_flat, &target_flat)?;
                
                total_final_loss += final_loss.to_scalar::<f32>()?;
                let final_grads = final_loss.backward()?;
                final_optimizer.step(&final_grads)?;

                // Calcular pérdida del ensemble consolidado para el log [B, L, V]
                let ens_logits_flat = ensemble_full_seq.reshape((batch_size * seq_length, vocab_size))?;
                total_ensemble_loss += candle_nn::loss::cross_entropy(&ens_logits_flat, &target_flat)?.to_scalar::<f32>()?;

                let current_steps = (batch_idx - start_batch + 1) as f32;
                let mut log_str = format!("\rEpoch {}/{} | Batch {}/{} | ", epoch+1, num_epochs, batch_idx, num_batches);
                for i in 0..num_models {
                    log_str.push_str(&format!("L{}={:.3} ", i+1, total_losses[i] / current_steps));
                }
                let duration = start_time.elapsed();
                log_str.push_str(&format!("| L_Ens={:.3} | L_Final={:.3} | {:?} ", total_ensemble_loss / current_steps, total_final_loss / current_steps, duration));
                print!("{}", log_str);
                io::stdout().flush()?;
                if batch_idx > start_batch && batch_idx % 50 == 0 {
                    println!("\nGuardando checkpoint en batch {}...", batch_idx);
                    for i in 0..num_models {
                        let _ = varmaps[i].save(&format!("luong_slstm_parallel_{}.safetensors", i + 1));
                    }
                    let _ = final_vm.save("luong_slstm_final.safetensors");
                    let seed = pick_seed(text_file, &tokenizer);
                    println!("  Seed: '{}'", seed);
                    let barrier_gen = Arc::new(Barrier::new(num_models));
                    let gen_sample = generate_threaded_text(&models, &final_model, &ln_comb, &drop_comb, &tokenizer, &seed, 120, &device, &barrier_gen)?;
                    println!("  Luong-sLSTM: {}\n", gen_sample);
                }
            }
        }
    }
    if train_mode {
        let seed = pick_seed(text_file, &tokenizer);
        let barrier = Arc::new(Barrier::new(num_models));
        let gen_final = generate_threaded_text(&models, &final_model, &ln_comb, &drop_comb, &tokenizer, &seed, 256, &device, &barrier)?;
        println!("{}", gen_final);
    } else {
        loop {
            print!("Seed (q para salir): ");
            io::stdout().flush()?;
            let mut seed = String::new();
            io::stdin().read_line(&mut seed)?;
            let s = seed.trim();
            if s.eq_ignore_ascii_case("q") { break; }
            let use_seed = if s.is_empty() { pick_seed(text_file, &tokenizer) } else { s.to_string() };
            print!("Longitud [Enter=256]: ");
            io::stdout().flush()?;
            let mut len_str = String::new();
            io::stdin().read_line(&mut len_str)?;
            let len = len_str.trim().parse::<usize>().unwrap_or(256);
            
            // Re-inicializamos el barrier para cada generación si es necesario
            let barrier = Arc::new(Barrier::new(num_models));
            let out = generate_threaded_text(&models, &final_model, &ln_comb, &drop_comb, &tokenizer, &use_seed, len, &device, &barrier)?;
            println!("{}", out);
        }
    }
    Ok(())
}

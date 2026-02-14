#![recursion_limit = "256"]

use candle_core::{Device, Tensor, DType};
use candle_nn::{VarBuilder, VarMap, Optimizer, AdamW, ParamsAdamW};
use anyhow::Result;
use std::fs::File;
use std::io::{self, Write, Read};
use std::path::Path;
use std::collections::{HashSet, VecDeque};
use std::time::Instant;
use std::sync::{Arc, Barrier};
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::AddedToken;

use xlstm::{LstmType, XLstm, XLstmconfig, LSTMState};
use rand::Rng;

pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_files(paths: &[&str], vocab_size: usize) -> Result<Self> {
        let special_tokens_strings = vec![
            "[EOS]".to_string(),
            "<|endoftext|>".to_string(),
        ];

        let special_tokens: Vec<AddedToken> = special_tokens_strings
            .iter()
            .map(|t| AddedToken::from(t, true))
            .collect();

        let model = BPE::builder()
            .byte_fallback(true)
            .build()
            .map_err(|e| anyhow::anyhow!(e))?;

        let mut tokenizer = HFTokenizer::new(model);

        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            ' ',
            PrependScheme::Always,
            true,
        )));

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
        tokenizer
            .train_from_files(&mut trainer_wrapper, files)
            .map_err(|e| anyhow::anyhow!(e))?;

        for token in special_tokens_strings {
            tokenizer.add_special_tokens(&[AddedToken::from(token, true)]);
        }

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<()> {
        self.tokenizer.save(path, true)
            .map_err(|e| anyhow::anyhow!("Error saving: {}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Error loading: {}", e))?;
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

struct StreamBatcher<'a> {
    file: File,
    tokenizer: &'a Tokenizer,
    buffer: VecDeque<usize>,
    chunk_size: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize,
    device: Device,
    total_size: u64,
    read_so_far: u64,
    done: bool,
}

impl<'a> StreamBatcher<'a> {
    fn new(path: &str, tokenizer: &'a Tokenizer, batch_size: usize, seq_length: usize, stride: usize, device: Device) -> Result<Self> {
        let file = File::open(path)?;
        let total_size = file.metadata()?.len();
        Ok(Self {
            file,
            tokenizer,
            buffer: VecDeque::new(),
            chunk_size: 1 << 20,
            batch_size,
            seq_length,
            stride,
            device,
            total_size,
            read_so_far: 0,
            done: false,
        })
    }

    fn need_tokens(&self) -> usize {
        (self.batch_size - 1) * self.stride + self.seq_length + 1
    }

    fn fill_buffer(&mut self) -> Result<()> {
        if self.done {
            return Ok(());
        }
        let mut buf = vec![0u8; self.chunk_size];
        let n = self.file.read(&mut buf)?;
        if n == 0 {
            self.done = true;
            return Ok(());
        }
        self.read_so_far += n as u64;
        let s = String::from_utf8_lossy(&buf[..n]);
        let tokens = self.tokenizer.encode(&s);
        for t in tokens {
            self.buffer.push_back(t);
        }
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<(Tensor, Tensor)>> {
        let needed = self.need_tokens();
        while self.buffer.len() < needed && !self.done {
            self.fill_buffer()?;
        }
        if self.buffer.len() < needed {
            return Ok(None);
        }
        let mut x_indices = Vec::with_capacity(self.batch_size * self.seq_length);
        let mut y_indices = Vec::with_capacity(self.batch_size * self.seq_length);
        for i in 0..self.batch_size {
            let base = i * self.stride;
            for j in 0..self.seq_length {
                let x = self.buffer[base + j] as u32;
                let y = self.buffer[base + j + 1] as u32;
                x_indices.push(x);
                y_indices.push(y);
            }
        }
        for _ in 0..(self.batch_size * self.stride) {
            self.buffer.pop_front();
        }
        let x = Tensor::from_vec(x_indices, (self.batch_size, self.seq_length), &self.device)?;
        let y = Tensor::from_vec(y_indices, (self.batch_size, self.seq_length), &self.device)?;
        Ok(Some((x, y)))
    }

    fn skip_batches(&mut self, n: usize) -> Result<()> {
        for _ in 0..n {
            if self.next_batch()?.is_none() {
                break;
            }
        }
        Ok(())
    }

    fn progress_pct(&self) -> f32 {
        if self.total_size == 0 {
            0.0
        } else {
            (self.read_so_far as f32 / self.total_size as f32) * 100.0
        }
    }
}

pub fn get_top_5_logits(logits: &Tensor) -> Result<Vec<(usize, f32)>> {
    let logits = logits.squeeze(0)?;
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;
    let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed.into_iter().take(5).collect())
}

fn sample_from_logits(averaged_logits: &Tensor, temperature: f32) -> Result<usize> {
    let scaled_logits = (averaged_logits / (temperature as f64))?;
    let probs = candle_nn::ops::softmax(&scaled_logits.squeeze(0)?, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;
    let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = 5;
    let top_k_probs = &indexed[..k.min(indexed.len())];
    let indices: Vec<usize> = top_k_probs.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = top_k_probs.iter().map(|(_, p)| *p).collect();
    let sum: f32 = weights.iter().sum();
    let mut rng = rand::rng();
    let mut sample: f32 = rng.random::<f32>() * sum;
    for (i, &p) in weights.iter().enumerate() {
        if sample <= p {
            return Ok(indices[i]);
        }
        sample -= p;
    }
    Ok(indices[0])
}

fn generate_threaded_text(
    models: &[XLstm],
    final_model: &XLstm,
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
    let mut current_states: Vec<Option<Vec<Option<LSTMState>>>> = vec![None; models.len()];
    let mut final_model_state: Option<Vec<Option<LSTMState>>> = None;
    let mut current_tokens = seed_tokens;
    for i in 0..length {
        let tokens_to_process = if i == 0 {
            current_tokens.clone()
        } else {
            vec![*current_tokens.last().unwrap()]
        };
        let seq_len = tokens_to_process.len();
        let indices_vec: Vec<u32> = tokens_to_process.iter().map(|&t| t as u32).collect();
        let input = Tensor::from_vec(indices_vec, (1, seq_len), device)?;
        let mut model_results = Vec::with_capacity(models.len());
        let scope_res: Result<()> = std::thread::scope(|s| {
            let mut handles = Vec::new();
            for (m_idx, model) in models.iter().enumerate() {
                let input_ref = &input;
                let state = current_states[m_idx].take();
                let b = barrier.clone();
                handles.push(s.spawn(move || {
                    b.wait();
                    let (output, next_state) = model.forward(input_ref, state)?;
                    let last_step_logits = output.narrow(1, seq_len - 1, 1)?.squeeze(1)?.detach();
                    b.wait();
                    Ok::<(Tensor, Vec<Option<LSTMState>>), anyhow::Error>((last_step_logits, next_state))
                }));
            }
            for handle in handles {
                let result = handle.join().unwrap()?;
                model_results.push(result);
            }
            Ok(())
        });
        scope_res?;
        let mut model_logits = Vec::with_capacity(model_results.len());
        for (m_idx, (logits, next_state)) in model_results.into_iter().enumerate() {
            model_logits.push(logits);
            current_states[m_idx] = Some(next_state.into_iter().map(|s| s.map(|state| state.detach())).collect());
        }
        let mut combined_logits = model_logits[0].clone();
        for j in 1..model_logits.len() {
            combined_logits = combined_logits.add(&model_logits[j])?;
        }
        let averaged_logits_ens = (combined_logits / (model_logits.len() as f64))?.flatten_all()?;
        let v_size = tokenizer.vocab_size();
        let mut residue_vec = vec![0.0f32; v_size];
        let last_token_id = current_tokens.last().copied().unwrap_or(0);
        residue_vec[last_token_id] = 1.0;
        let residue = Tensor::from_vec(residue_vec, (v_size,), device)?;
        let combined_input = averaged_logits_ens.add(&residue)?.unsqueeze(0)?.unsqueeze(0)?;
        let (final_output, next_final_state) = final_model.forward(&combined_input, final_model_state)?;
        final_model_state = Some(next_final_state.into_iter().map(|s| s.map(|state| state.detach())).collect());
        let final_logits = final_output.squeeze(0)?.squeeze(0)?;
        let next_token = sample_from_logits(&final_logits, 0.9)?;
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin tinystory3v -- <archivo.txt>");
        std::process::exit(1);
    }
    let text_file = &args[1];
    let tokenizer_path_v3 = "tinystory3v_tokenizer.json";
    let tokenizer_path_v2 = "tinystory2v_tokenizer.json";
    let target_vocab_size = 1024;
    let device = Device::Cpu;

    let tokenizer = if Path::new(tokenizer_path_v3).exists() {
        Tokenizer::load(tokenizer_path_v3)?
    } else if Path::new(tokenizer_path_v2).exists() {
        Tokenizer::load(tokenizer_path_v2)?
    } else {
        let tokenizer = Tokenizer::from_files(&[text_file], target_vocab_size)?;
        tokenizer.save(tokenizer_path_v3)?;
        tokenizer
    };

    let vocab_size = tokenizer.vocab_size();
    let hidden_size = 256;
    let num_layers = 1;
    let num_blocks = 1;
    let output_size = vocab_size;
    let seq_length = 160;
    let batch_size = 20;
    let stride = 160;
    let num_epochs = 20;

    let config = XLstmconfig::new(hidden_size, hidden_size, num_layers, num_blocks, output_size)
        .with_vocab_size(vocab_size)
        .with_lstm_type(LstmType::SLSTM)
        .with_use_projection(true);

    let num_models = 3;
    let mut models = Vec::new();
    let mut varmaps = Vec::new();
    let mut optimizers = Vec::new();

    for i in 1..=num_models {
        let vm = VarMap::new();
        let vb = VarBuilder::from_varmap(&vm, DType::F32, &device);
        let model = config.init(vb)?;
        let mut params = Vec::new();
        {
            let data = vm.data().lock().unwrap();
            for (_, var) in data.iter() {
                params.push(var.clone());
            }
        }
        let optimizer = AdamW::new(params, ParamsAdamW { lr: 2.5e-4, ..Default::default() })?;
        models.push(model);
        varmaps.push(vm);
        optimizers.push(optimizer);
        let m_path_3v = format!("tinystory3v_parallel_{}.safetensors", i);
        let m_path_2v = format!("tinystory2v_parallel_{}.safetensors", i);
        let m_path_base = format!("tinystory_cortex_{}.safetensors", i);
        if Path::new(&m_path_3v).exists() {
            varmaps[i-1].load(&m_path_3v)?;
        } else if Path::new(&m_path_2v).exists() {
            varmaps[i-1].load(&m_path_2v)?;
        } else if Path::new(&m_path_base).exists() {
            varmaps[i-1].load(&m_path_base)?;
        }
    }

    let final_config = XLstmconfig::new(vocab_size, 320, num_layers, 2, output_size)
        .with_lstm_type(LstmType::SLSTM)
        .with_use_projection(true);
    let mut final_vm = VarMap::new();
    let final_vb = VarBuilder::from_varmap(&final_vm, DType::F32, &device);
    let final_model = final_config.init(final_vb)?;
    let mut final_params = Vec::new();
    {
        let data = final_vm.data().lock().unwrap();
        for (_, var) in data.iter() {
            final_params.push(var.clone());
        }
    }
    let mut final_optimizer = AdamW::new(final_params, ParamsAdamW { lr: 4e-4, ..Default::default() })?;
    let final_model_path = "tinystory3v_final.safetensors";
    let mut final_available = false;
    if Path::new(final_model_path).exists() {
        if final_vm.load(final_model_path).is_ok() {
            final_available = true;
        }
    }
    if !final_available && Path::new("tinystory2v_final.safetensors").exists() {
        if final_vm.load("tinystory2v_final.safetensors").is_ok() {
            final_available = true;
        }
    }

    let mut all_models_exist = true;
    for i in 1..=num_models {
        let has_3v = Path::new(&format!("tinystory3v_parallel_{}.safetensors", i)).exists();
        let has_2v = Path::new(&format!("tinystory2v_parallel_{}.safetensors", i)).exists();
        let has_base = Path::new(&format!("tinystory_cortex_{}.safetensors", i)).exists();
        if !has_3v && !has_2v && !has_base {
            all_models_exist = false;
            break;
        }
    }
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
        let thread_barrier = Arc::new(Barrier::new(num_models));
        let train_mask = vec![true; num_models];
        let mut start_batch = 5310;//3700 //4200
        for epoch in 0..num_epochs {
            let mut total_losses = vec![0.0f32; num_models];
            let mut total_ensemble_loss = 0.0f32;
            let mut total_final_loss = 0.0f32;
            let start_time = Instant::now();
            if epoch > 0 { start_batch = 0; }
            let mut loader = StreamBatcher::new(text_file, &tokenizer, batch_size, seq_length, stride, device.clone())?;
            loader.skip_batches(start_batch)?;
            let mut batch_idx: usize = start_batch;
            loop {
                let next = loader.next_batch()?;
                if next.is_none() {
                    break;
                }
                let (input_batch, target_batch) = next.unwrap();
                let target_flat = target_batch.reshape((batch_size * seq_length,))?;
                let input_ref = &input_batch;
                let target_ref = &target_flat;
                std::thread::scope(|s| {
                    let mut handles = Vec::new();
                    let zip_iter = models.iter().zip(optimizers.iter_mut());
                    for (m_idx, (model, optimizer)) in zip_iter.enumerate() {
                        let b = &thread_barrier;
                        let should_train = train_mask[m_idx] && !train_only_final;
                        handles.push(s.spawn(move || {
                            b.wait();
                            let (logits, _) = model.forward(input_ref, None)?;
                            let logits_flat = logits.reshape((batch_size * seq_length, vocab_size))?;
                            let loss = candle_nn::loss::cross_entropy(&logits_flat, target_ref)?;
                            let loss_val = loss.to_scalar::<f32>()?;
                            if should_train {
                                let grads = loss.backward()?;
                                optimizer.step(&grads)?;
                            }
                            b.wait();
                            Ok::<(f32, Tensor), anyhow::Error>((loss_val, logits_flat.detach()))
                        }));
                    }
                    let mut batch_logits = Vec::new();
                    for (m_idx, handle) in handles.into_iter().enumerate() {
                        let (loss_val, logits_val) = handle.join().unwrap().unwrap();
                        total_losses[m_idx] += loss_val;
                        batch_logits.push(logits_val);
                    }
                    let mut combined_logits = batch_logits[0].clone();
                    for j in 1..batch_logits.len() {
                        combined_logits = combined_logits.add(&batch_logits[j]).unwrap();
                    }
                    let averaged_logits_ens = (combined_logits / (batch_logits.len() as f64)).unwrap();
                    let input_ref_flat = input_ref.flatten_all().unwrap();
                    let mut one_hot_data = vec![0.0f32; batch_size * seq_length * vocab_size];
                    let input_indices = input_ref_flat.to_vec1::<u32>().unwrap();
                    for (i, &idx) in input_indices.iter().enumerate() {
                        one_hot_data[i * vocab_size + (idx as usize)] = 1.0;
                    }
                    let residue_flat = Tensor::from_vec(one_hot_data, (batch_size * seq_length, vocab_size), &device).unwrap();
                    let combined_input_flat = averaged_logits_ens.add(&residue_flat).unwrap();
                    let combined_input = combined_input_flat.reshape((batch_size, seq_length, vocab_size)).unwrap();
                    let (final_output, _) = final_model.forward(&combined_input, None).unwrap();
                    let final_logits_flat = final_output.reshape((batch_size * seq_length, vocab_size)).unwrap();
                    let final_loss = candle_nn::loss::cross_entropy(&final_logits_flat, target_ref).unwrap();
                    let final_loss_val = final_loss.to_scalar::<f32>().unwrap();
                    total_final_loss += final_loss_val;
                    let ensemble_loss = candle_nn::loss::cross_entropy(&averaged_logits_ens, target_ref).unwrap();
                    total_ensemble_loss += ensemble_loss.to_scalar::<f32>().unwrap();
                    let final_grads = final_loss.backward().unwrap();
                    final_optimizer.step(&final_grads).unwrap();
                });
                let current_steps = (batch_idx - start_batch + 1) as f32;
                let mut log_str = format!("\rEpoch {}/{} | Batch {} | Progreso {:.1}% | ", epoch+1, num_epochs, batch_idx, loader.progress_pct());
                for i in 0..num_models {
                    log_str.push_str(&format!("L{}={:.3} ", i+1, total_losses[i] / current_steps));
                }
                log_str.push_str(&format!("| L_Ens={:.3} | L_Final={:.3}", total_ensemble_loss / current_steps, total_final_loss / current_steps));
                print!("{}", log_str);
                io::stdout().flush()?;
                if batch_idx > start_batch && batch_idx % 15 == 0 {
                    println!("\nGuardando checkpoint en batch {}...", batch_idx);
                    for i in 0..num_models { varmaps[i].save(&format!("tinystory3v_parallel_{}.safetensors", i+1))?; }
                    final_vm.save(final_model_path)?;
                    let seed = pick_seed(text_file, &tokenizer);
                    println!("  Seed: '{}'", seed);
                    let gen_sample = generate_threaded_text(&models, &final_model, &tokenizer, &seed, 500, &device, &thread_barrier)?;
                    println!("  TinyStory3V: {}\n", gen_sample);
                }
                batch_idx += 1;
            }
            println!("\nEpoch {} OK ({:.1}s). Guardando...", epoch+1, start_time.elapsed().as_secs_f32());
            for i in 0..num_models { varmaps[i].save(&format!("tinystory3v_parallel_{}.safetensors", i+1))?; }
            final_vm.save(final_model_path)?;
            let seed = pick_seed(text_file, &tokenizer);
            println!("  Seed: '{}'", seed);
            let gen_sample = generate_threaded_text(&models, &final_model, &tokenizer, &seed, 300, &device, &thread_barrier)?;
            println!("  TinyStory3V: {}\n", gen_sample);
        }
    }

    let thread_barrier = Arc::new(Barrier::new(num_models));
    println!("\n--- Modo Interactivo (TinyStory3V) ---");
    println!("Comandos:");
    println!("  - Escribe 'len N' para cambiar la longitud de generación");
    println!("  - Escribe 'exit' para salir");
    let mut gen_length: usize = 100;
    loop {
        print!("Input (len={}) > ", gen_length);
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input.is_empty() { continue; }
        if input == "exit" { break; }
        if input.to_lowercase().starts_with("len") {
            let parts: Vec<&str> = input.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(n) = parts[1].parse::<usize>() {
                    gen_length = n;
                    println!("Nueva longitud de generación: {} tokens\n", gen_length);
                }
            }
            continue;
        }
        let gen_interactive = generate_threaded_text(&models, &final_model, &tokenizer, input, gen_length, &device, &thread_barrier)?;
        println!("TinyStory3V (3 Parallel + 2 Block Final): {}\n", gen_interactive);
    }
    Ok(())
}

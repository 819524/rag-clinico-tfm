# Guía de optimización de vLLM (voz05 · RTX PRO 6000 Blackwell)

Referencia práctica de los parámetros de `vllm serve` para exprimir el rendimiento, con
nuestros valores actuales y las palancas por explorar. Versión: **vLLM 0.23.0**.
Lista COMPLETA de flags (268): `VLLM_ENGINE_ARGS_full.txt` · online: <https://docs.vllm.ai/en/stable/configuration/optimization/>.

> Ayuda local: `vllm serve --help` (por secciones) · `--help=all` (todo) · `--help=CacheConfig` (una sección) · `--help=max-model-len` (un flag).

---

## Config actual (baseline, la que ganó el benchmark)
```bash
vllm serve <modelo-AWQ-4bit> --port 8000 --served-model-name gemma4-awq \
  --gpu-memory-utilization 0.9 --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-batched-tokens 16384
```
Ya activo por defecto: PagedAttention · continuous batching · **CUDA graphs + torch.compile** (nivel `-O2`) · **prefix caching** · chunked prefill · FlashInfer.

---

## Parámetros por categoría (los que importan)

### Memoria / KV-caché — `CacheConfig`
| Flag | Nuestro valor | Efecto |
|---|---|---|
| `--gpu-memory-utilization` | 0.9 | % de VRAM para modelo + pool de KV. Subir a 0.95/0.97 = más KV. |
| `--kv-cache-dtype` | auto (bf16) | **`fp8`** → halva la memoria de KV (más concurrencia/contexto). También admite `nvfp4`. |
| `--kv-cache-memory-bytes` | — | Fija el tamaño del pool de KV en bytes (control fino). |
| `--max-model-len` | 16384 | Contexto máx por petición. A menor, más peticiones caben en KV. |
| `--cpu-offload-gb` | — | Descarga pesos a RAM si falta VRAM (más lento). |

### Throughput / batching — `SchedulerConfig`
| Flag | Nuestro valor | Efecto |
|---|---|---|
| `--max-num-batched-tokens` | 16384 | Tokens por iteración; >8192 recomendado, clave en prefill RAG (prompts largos). |
| `--max-num-seqs` | (auto) | Nº máx de secuencias concurrentes. Subir si hay VRAM y no satura. |
| `--enable-chunked-prefill` | on | Intercala prefill/decode (mejor bajo carga mixta). |
| `--enable-prefix-caching` | on | Reutiliza KV del prefijo compartido → **decisivo en RAG** (nuestro A/B: 2.5×). |
| `--long-prefill-token-threshold` | — | Umbral para trocear prefills largos. |

### Latencia / compilación — `CompilationConfig`
| Flag | Nuestro valor | Efecto |
|---|---|---|
| `-O0/-O1/-O2/-O3` (`--compilation-config`) | -O2 (def) | Compromiso arranque↔rendimiento. `-O3` = más fusiones/CUDA graphs (más lento de arrancar). |
| `--enforce-eager` | off | Desactiva CUDA graphs. Solo debug; **dejarlo off**. |
| `-cc`/`--compilation-config` | — | Control fino: `cudagraph_mode`, `cudagraph_capture_sizes`, `inductor_*`. |
| `--speculative-config` (`-sc`) | — | **Decodificación especulativa** (n-gram/draft/EAGLE) → baja TTFT/TPOT. |

### Paralelismo (multi-GPU) — `ParallelConfig`
| Flag | Efecto |
|---|---|
| `--tensor-parallel-size 2` (`-tp`) | Reparte el modelo en las 2 Blackwell → menor latencia por token, más KV. |
| `--pipeline-parallel-size` | Por capas (multi-nodo, no aplica en 1 nodo). |
| `--data-parallel-size` | Réplicas para throughput (varias instancias). |

### Cuantización / kernels — `ModelConfig`
| Flag | Efecto |
|---|---|
| `--quantization` (`-q`) | Formato de pesos (autodetectado del modelo). **AWQ vs NVFP4** cambia los kernels. |
| `--dtype` | Precisión de activaciones (auto = bf16). |

---

## Experimentos de optimización propuestos (comandos)

### 1 · NVFP4 — techo del hardware Blackwell ⭐
Descargar `nvidia/Gemma-4-26B-A4B-NVFP4` y servir (usa tensor cores FP4 nativos):
```bash
vllm serve ~/llm-bench/models/hf/Gemma-4-26B-A4B-NVFP4 \
  --port 8000 --served-model-name gemma4-awq \
  --gpu-memory-utilization 0.9 --max-model-len 16384 \
  --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-batched-tokens 16384
```

### 2 · Decodificación especulativa (n-gram, sin modelo draft)
```bash
  ... --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":8}'
```

### 3 · Tensor-parallel en las 2 GPUs (latencia)
```bash
  CUDA_VISIBLE_DEVICES=0,1 vllm serve ... --tensor-parallel-size 2
```
(Habría que mover los embeddings fuera de GPU1.)

### 4 · KV-caché FP8 (más margen de concurrencia/contexto)
```bash
  ... --kv-cache-dtype fp8
```

### 5 · Compilación agresiva
```bash
  ... -O3        # más fusiones y CUDA graphs (arranque más lento, decode algo mejor)
```

---

## RESULTADOS: Decodificación especulativa (MEDIDO, A/B x10) ⭐

Conclusión: **aceptación alta ≠ speedup garantizado**. El speculative solo compensa si el modelo es
realmente *memory-bound* (hay cómputo GPU ocioso que llenar). Tres casos medidos en voz05:

| Modelo | Método | Aceptación | Veredicto (throughput) | Por qué |
|---|---|---|---|---|
| **gemma-4 26B-A4B** (MoE) | n-gram | 26% | **−10 a −23%** ✗ | MoE activa ~4B/token → poco cómputo ocioso; + n-gram solo caza texto verbatim (RAG parafrasea) |
| **gemma-4 12B** (denso puro) | MTP (assistant) | 60% | **+4 a +10%** ✓ | denso = memory-bound → hay hueco; drafter oficial acierta mucho |
| **Qwen3.6-27B** (híbrido Mamba) | MTP (nextn) | **91%** | **marginal** (+5% single, −8% @25u) | híbrido Mamba no es tan memory-bound + overhead del drafter se come la ganancia |

Datos clave:
- **n-gram (`method:ngram`)**: el "drafter" es una búsqueda de texto en el prompt (sin modelo). Barato pero
  solo acierta lo verbatim. En RAG médico (respuestas parafraseadas) → 26% → **resta** en el 26B-MoE.
- **MTP / draft-model**: un drafter entrenado (gemma `-assistant` de 4 capas, o el nextn embebido de Qwen).
  Mucha más aceptación. gemma 12B AWQ + `--speculative-config '{"method":"mtp","model":"…-assistant","num_speculative_tokens":4}'` → limpio, +4-10%.
- **Qwen3.6-27B** (`hampsonw/…AWQ-BF16-INT4-mtp-bf16`, MTP bf16, `method:mtp,num_speculative_tokens:1`): 91% aceptación
  pero solo +5% single-stream. Es **híbrido Mamba+atención** → la premisa "hay cómputo ocioso" se debilita.
  Nota de arranque: baja `--max-num-seqs` (≤ nº de bloques de caché Mamba) o falla el CUDA graph capture.
- **Modelo pensante**: Qwen3.6 vuelca el CoT como respuesta → desactivar con env `GENERATOR_DISABLE_THINKING=1`
  (gated en `fase4_generator._stream_openai`: añade `chat_template_kwargs={"enable_thinking":false}`).

Contexto de throughput @25u (mediana): **gemma 26B-A4B 709 t/s** ≫ Qwen3.6-27B 186 t/s (≈4×, dense vs MoE activa).
→ Para servir, el 26B-MoE gana por goleada; el speculative **no cambia la elección**. Mejor config sigue siendo
**vLLM 26B-A4B AWQ sin speculative**.

---

## Enlaces
- Optimización y tuning: <https://docs.vllm.ai/en/stable/configuration/optimization/>
- Todos los engine args: <https://docs.vllm.ai/en/stable/configuration/engine_args/>
- Referencia local completa: `VLLM_ENGINE_ARGS_full.txt` (268 flags de esta build 0.23.0).

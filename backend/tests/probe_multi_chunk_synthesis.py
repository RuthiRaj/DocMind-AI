import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

from app.core.prompts import RAG_SYSTEM_PROMPT

# Realistic 7 multi-chunk context describing the Transformer architecture across different pages
chunks = [
    "Chunk chunk_001 (Page 1):\nThe Transformer is the first sequence transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. The Transformer allows for significantly more parallelization and was trained on 8 P100 GPUs.",
    "Chunk chunk_002 (Page 2):\nEncoder and Decoder Stacks: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism, and a simple, position-wise fully connected feed-forward network. Residual connections are employed around each sub-layer, followed by layer normalization.",
    "Chunk chunk_003 (Page 2):\nThe decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. We modify the self-attention sub-layer in the decoder to prevent positions from attending to subsequent positions (masking).",
    "Chunk chunk_004 (Page 3):\nAttention: An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. Scaled Dot-Product Attention computes: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V.",
    "Chunk chunk_005 (Page 4):\nMulti-Head Attention: Instead of performing a single attention function with d_model-dimensional keys, values and queries, we linearly project queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions. We compute attention in parallel and concatenate them.",
    "Chunk chunk_006 (Page 5):\nPosition-wise Feed-Forward Networks: In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network: FFN(x) = max(0, xW1 + b1)W2 + b2. It consists of two linear transformations with a ReLU activation in between.",
    "Chunk chunk_007 (Page 8):\nPositional Encoding: Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute positions of the tokens in the sequence: PE(pos, 2i) = sin(pos/10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))."
]

context = "\n\n".join(chunks)
question = "Explain the complete Transformer architecture."

messages = [
    {"role": "system", "content": RAG_SYSTEM_PROMPT},
    {"role": "user", "content": f"DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION: {question}"}
]

print(">>> RUNNING MULTI-CHUNK SYNTHESIS QUERY AGAINST openai/gpt-oss-20b <<<")
for max_t in [512, 1024, 1536]:
    print(f"\n==================== TESTING max_tokens = {max_t} ====================")
    raw = client.chat.completions.with_raw_response.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        max_tokens=max_t,
        temperature=0.2
    )
    parsed = raw.parse()
    choice = parsed.choices[0]
    msg = choice.message
    reasoning = getattr(msg, "reasoning", "") or ""
    content = msg.content or ""
    usage = parsed.usage
    r_tokens = usage.completion_tokens_details.reasoning_tokens if usage and usage.completion_tokens_details else "N/A"
    
    print(f"finish_reason: {choice.finish_reason}")
    print(f"prompt_tokens: {usage.prompt_tokens}")
    print(f"completion_tokens: {usage.completion_tokens} (reasoning_tokens: {r_tokens})")
    print(f"content length: {len(content)} chars")
    print(f"content preview:\n{content[:300]}")
    print(f"reasoning length: {len(reasoning)} chars")
    print(f"reasoning preview:\n{reasoning[:200]}...")

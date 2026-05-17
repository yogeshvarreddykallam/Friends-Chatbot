import streamlit as st
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
import os
import base64

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Central Perk Chat",
    page_icon="☕",
    layout="wide"
)

# --- 2. BACKGROUND IMAGE LOADER ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Make text readable over image */
        .stMarkdown, .stTitle, h1, h2, h3, h4, p {{
            text-shadow: 2px 2px 4px #000000;
            color: #ffffff !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Try to set background if file exists
if os.path.exists("assets/friends_bg.jpg"):
    set_background("assets/friends_bg.jpg")

# --- 3. CUSTOM STYLING ---
st.markdown("""
<style>
    /* Header Title */
    .title-text {
        font-family: 'Courier New', Courier, monospace;
        color: #ff9933 !important; /* Orange */
        text-align: center;
        font-size: 70px;
        font-weight: bold;
        text-shadow: 4px 4px 8px #000000;
        margin-bottom: 5px;
        background-color: rgba(0, 0, 0, 0.6); /* Semi-transparent box */
        border-radius: 15px;
        padding: 10px;
    }
    
    /* Chat Bubbles */
    .stChatMessage {
        background-color: rgba(46, 46, 46, 0.9); /* Semi-transparent dark */
        border: 2px solid #ff9933;
        border-radius: 15px;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #261a12;
    }
    
    /* Avatar Size */
    .stChatMessage .st-emotion-cache-1p1m4t5 {
        width: 3.5rem;
        height: 3.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title-text">☕ CENTRAL PERK AI</div>', unsafe_allow_html=True)

# --- 4. SIDEBAR ---
st.sidebar.title("Select a Friend")
char_choice = st.sidebar.radio(
    "Who is on the orange couch?",
    ("Joey", "Chandler", "Monica", "Ross", "Rachel")
)

# --- CHARACTER CONFIGURATION (UPDATED FILENAMES) ---
char_map = {
    "Joey":     {"emoji": "🍕", "folder": "joey",     "image": "assets/joey.jpg"},  # JPG!
    "Chandler": {"emoji": "🦆", "folder": "chandler", "image": "assets/chandler.png"},
    "Monica":   {"emoji": "👩🏻‍🍳", "folder": "monica",   "image": "assets/monica.png"},
    "Ross":     {"emoji": "🦖", "folder": "ross",     "image": "assets/ross.png"},
    "Rachel":   {"emoji": "🛍️", "folder": "rachel",   "image": "assets/rachel.png"}
}

current_char_data = char_map[char_choice]

# Helper to safely get the avatar
def get_avatar(char_name):
    data = char_map[char_name]
    if os.path.exists(data["image"]):
        return data["image"]
    return data["emoji"]

# --- 5. MODEL LOADER ---
def find_best_model_path(base_folder, char_name):
    target_path = os.path.join(base_folder, char_name)
    if not os.path.exists(target_path): return None

    files = os.listdir(target_path)
    if "model.safetensors" in files or "pytorch_model.bin" in files:
        return target_path 

    checkpoints = [f for f in files if f.startswith("checkpoint-")]
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('-')[1]), reverse=True)
        return os.path.join(target_path, checkpoints[0])
    return None

@st.cache_resource
def load_model(character_folder):
    base_path = "friends" 
    final_path = find_best_model_path(base_path, character_folder)
    
    if final_path is None: return None
    
    print(f"Loading {character_folder} from: {final_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(final_path)
    model = GPT2LMHeadModel.from_pretrained(final_path)
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return pipeline('text-generation', model=model, tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)

# Load Model
with st.spinner(f"Getting {char_choice} some coffee..."):
    generator = load_model(current_char_data["folder"])

if generator is None:
    st.error(f"❌ Error: Model for {char_choice} not found.")
    st.stop()

# --- 6. CHAT INTERFACE ---

if "last_char" not in st.session_state: st.session_state.last_char = char_choice
if st.session_state.last_char != char_choice:
    st.session_state.messages = []
    st.session_state.last_char = char_choice

if "messages" not in st.session_state: st.session_state.messages = []

# Display History
for message in st.session_state.messages:
    if message["role"] == "assistant":
        avatar_to_show = message.get("avatar_path", get_avatar(char_choice))
    else:
        avatar_to_show = None 

    with st.chat_message(message["role"], avatar=avatar_to_show):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input(f"Talk to {char_choice}..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    current_avatar = get_avatar(char_choice)
    
    with st.chat_message("assistant", avatar=current_avatar):
        message_placeholder = st.empty()
        formatted_prompt = f"User: {prompt}\n{char_choice}:"
        
        try:
            result = generator(formatted_prompt, max_length=150, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.9, do_sample=True, truncation=True)
            full_text = result[0]['generated_text']
            reply = full_text.split(f"{char_choice}:")[-1].split("User:")[0].split("<|endoftext|>")[0].strip()
        except:
            reply = "..."

        message_placeholder.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply, "avatar_path": current_avatar})
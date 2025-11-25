import streamlit as st
import chess
import chess.svg
import base64
import torch
import os
import time
import re
import mlflow
from chess_rl import ChessNet, DQNAgent, train_dqn_generator, ReplayBuffer

# 1. SETUP
st.set_page_config(page_title="Chess AI Lab", layout="wide")
MLFLOW_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URI)

st.markdown("""
<style>
    .chess-board { display: flex; justify_content: center; margin-bottom: 20px; }
    .success-box { padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px; text-align: center; font-weight: bold;}
    .metric-box { padding: 10px; border-radius: 5px; text-align: center; background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

def render_svg(board, size=500):
    svg = chess.svg.board(board=board, size=size)
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    return f'<div class="chess-board"><img src="data:image/svg+xml;base64,{b64}" width="{size}" height="{size}"/></div>'

st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.radio("Select Mode:", ["Play vs AI", "Train AI (Visual)"])

# ==========================================
# MODE 1: PLAY VS AI (SMART)
# ==========================================
if mode == "Play vs AI":
    st.title("♟️ Play vs Your Models")
    
    # 1. Search Recursive for .pth
    search_path = os.getcwd() 
    st.sidebar.markdown(f"📂 **Scanning:** `{search_path}`")
    
    found_models = []
    
    for root, dirs, files in os.walk(search_path):
        if any(part.startswith('.') for part in root.split(os.sep)): continue
            
        for file in files:
            if file.endswith(".pth"):
                full_path = os.path.join(root, file)
                mod_time = os.path.getmtime(full_path)
                
                # --- EXTRACT EPISODES FROM FILENAME ---
                match = re.search(r'(\d+)ep', file)
                if match:
                    eps = match.group(1)
                    display_name = f"🧠 AI (Trained {eps} Eps) - {file}"
                else:
                    display_name = f"📁 {file}"

                found_models.append((display_name, full_path, mod_time))
    
    if not found_models:
        st.warning("⚠️ No models found.")
        st.info("Go to 'Train AI' and run for 5 episodes.")
    else:
        # Sort by newest
        found_models.sort(key=lambda x: x[2], reverse=True)
        
        # Dropdown
        model_options = {path: name for name, path, mtime in found_models}
        selected_path = st.sidebar.selectbox("Select Opponent:", list(model_options.keys()), format_func=lambda x: model_options[x])
        
        if st.sidebar.button("Load this Brain 🧠"):
            st.session_state.ai_model_path = selected_path
            st.session_state.board = chess.Board() # Reset game
            st.success(f"Loaded: {os.path.basename(selected_path)}")

    # --- Game Logic ---
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()

    ai_agent = None
    if "ai_model_path" in st.session_state:
        try:
            # Check format: Dict vs Module
            checkpoint = torch.load(st.session_state.ai_model_path)
            if isinstance(checkpoint, dict):
                model = ChessNet()
                model.load_state_dict(checkpoint)
            elif isinstance(checkpoint, torch.nn.Module):
                model = checkpoint
            else:
                model = None
                
            if model:
                model.eval()
                ai_agent = DQNAgent(model, ReplayBuffer(1))
        except Exception as e:
            st.error(f"Error loading model: {e}")

    col1, col2 = st.columns([1.5, 1])
    
    with col2:
        st.markdown("### Game Controls")
        if ai_agent:
            st.markdown('<div class="success-box">🟢 AI is Active</div>', unsafe_allow_html=True)
        else:
            st.warning("🔴 AI not loaded")
            
        move_input = st.text_input("Your Move (e.g., 'e4' or 'e2e4'):", key="move_input")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("Make Move", type="primary"):
                board = st.session_state.board
                try:
                    # 1. Try Human Notation (e4)
                    try:
                        move = board.parse_san(move_input)
                    except:
                        # 2. Try Computer Notation (e2e4)
                        move = chess.Move.from_uci(move_input)

                    if move in board.legal_moves:
                        board.push(move)
                        if not board.is_game_over() and ai_agent:
                            with st.spinner("AI is Thinking..."):
                                ai_move = ai_agent.choose_move(board, epsilon=0.0)
                                board.push(ai_move)
                        st.rerun()
                    else:
                        st.error("Illegal move.")
                except:
                    st.error("Invalid move format.")
        
        with col_btn2:
            if st.button("Reset Game"):
                st.session_state.board = chess.Board()
                st.rerun()

    with col1:
        st.markdown(render_svg(st.session_state.board), unsafe_allow_html=True)

# ==========================================
# MODE 2: TRAIN AI (VISUAL)
# ==========================================
elif mode == "Train AI (Visual)":
    st.title("🧠 AI Training Room")
    st.info(f"Logging to: {MLFLOW_URI}")
    
    col1, col2, col3 = st.columns(3)
    with col1: episodes = st.slider("Total Episodes", 5, 2000, 10)
    with col2: speed = st.slider("Speed", 0.0, 0.5, 0.01)
    with col3: start_btn = st.button("▶️ Start Training", type="primary")

    board_col, stats_col = st.columns([1.2, 1])
    with board_col: board_placeholder = st.empty()
    with stats_col: 
        progress_bar = st.progress(0)
        status_text = st.empty()
        q_metric = st.empty()
        log_box = st.empty()

    if start_btn:
        trainer = train_dqn_generator(episodes)
        
        for item in trainer:
            board, ep, move_num, info = item
            
            if ep == -1:
                st.balloons()
                st.success("Training Complete!")
                break
            
            board_placeholder.markdown(render_svg(board, size=450), unsafe_allow_html=True)
            pct = ep / episodes
            progress_bar.progress(pct)
            status_text.text(f"Episode: {ep}/{episodes} | Move: {move_num}")
            
            q = info.get('q_val', 0)
            q_color = "green" if q > 0 else "red"
            q_metric.markdown(f"**Q-Value:** <span style='color:{q_color}'>{q:.4f}</span>", unsafe_allow_html=True)
            
            winner = info.get('winner')
            if winner:
                log_box.markdown(f"<div class='success-box'>Ep {ep} Winner: {winner}</div>", unsafe_allow_html=True)
                if speed > 0: time.sleep(0.5)
            
            if speed > 0: time.sleep(speed)
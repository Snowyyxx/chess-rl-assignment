import os
import random
from collections import deque

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch

# ==========================================================
# 0. DEVICE + SEEDS
# ==========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ==========================================================
# 1. THE NETWORK (Value Function V(s))
# ==========================================================
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)          # (B, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        return self.fc2(x)                 # (B, 1) unbounded scalar V(s)


# ==========================================================
# 2. HELPER: Board to Tensor
#    12 channels: (White pieces, then Black pieces)
# ==========================================================
def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board into a (1, 12, 8, 8) float32 tensor.
    Channel order:
    0-5: White (P, N, B, R, Q, K)
    6-11: Black (P, N, B, R, Q, K)
    """
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
              chess.ROOK, chess.QUEEN, chess.KING]
    matrix = np.zeros((12, 8, 8), dtype=np.float32)

    for i, piece_type in enumerate(pieces):
        for sq in board.pieces(piece_type, chess.WHITE):
            r, c = divmod(sq, 8)
            matrix[i, 7 - r, c] = 1.0
        for sq in board.pieces(piece_type, chess.BLACK):
            r, c = divmod(sq, 8)
            matrix[i + 6, 7 - r, c] = 1.0

    return torch.tensor(matrix).unsqueeze(0)  # (1, 12, 8, 8)


# ==========================================================
# 3. REPLAY BUFFER
# ==========================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        # Store as CPU tensors + scalars for memory efficiency
        self.buffer.append((
            state.cpu(),
            float(reward),
            next_state.cpu(),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states, dim=0)       # (B, 12, 8, 8)
        next_states = torch.cat(next_states, 0)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ==========================================================
# 4. DQN-LIKE AGENT FOR VALUE FUNCTION
# ==========================================================
class DQNAgent:
    def __init__(
        self,
        model: ChessNet,
        buffer: ReplayBuffer,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update_steps: int = 500,
    ):
        self.device = DEVICE

        self.model = model.to(self.device)
        self.target_model = ChessNet().to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.buffer = buffer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma

        self.target_update_steps = target_update_steps
        self.train_steps = 0

    def _update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_value(self, board: chess.Board) -> float:
        self.model.eval()
        with torch.no_grad():
            state = board_to_tensor(board).to(self.device)
            v = self.model(state).item()
        self.model.train()
        return v

    def choose_move(self, board: chess.Board, epsilon: float = 0.1):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        # ε-greedy
        if random.random() < epsilon:
            return random.choice(legal_moves)

        best_move = None
        best_value = -float("inf")

        self.model.eval()
        with torch.no_grad():
            for move in legal_moves:
                board.push(move)
                s_next = board_to_tensor(board).to(self.device)
                # model(s_next) is for side-to-move in next position
                v_next = self.model(s_next).item()
                # from current mover's perspective, value is -v_next
                v_current = -v_next
                board.pop()

                if v_current > best_value:
                    best_value = v_current
                    best_move = move
        self.model.train()
        return best_move

    def optimize(self, batch_size: int = 32):
        if len(self.buffer) < batch_size:
            return 0.0

        states, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_vals = self.target_model(next_states)  # (B,1)
            # value from previous mover's perspective is -next_vals
            targets = rewards + (1.0 - dones) * self.gamma * (-next_vals)

        preds = self.model(states)
        loss = self.loss_fn(preds, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_steps == 0:
            self._update_target_network()

        return loss.item()


# ==========================================================
# 5. HELPER: MATERIAL VALUE (FROM GIVEN COLOR'S VIEW)
# ==========================================================
def get_material_value(board: chess.Board, color: bool) -> int:
    """
    Material score from 'color's perspective.
    Positive => advantage for 'color'.
    """
    values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
    }
    score = 0
    opp = not color
    for piece_type, v in values.items():
        score += len(board.pieces(piece_type, color)) * v
        score -= len(board.pieces(piece_type, opp)) * v
    return score


# ==========================================================
# 6. EVALUATION: AGENT (WHITE) VS RANDOM (BLACK)
# ==========================================================
def play_vs_random(agent: DQNAgent, n_games: int = 5, max_moves: int = 80):
    wins = draws = losses = 0

    for _ in range(n_games):
        board = chess.Board()
        moves = 0

        while not board.is_game_over() and moves < max_moves:
            if board.turn == chess.WHITE:
                move = agent.choose_move(board, epsilon=0.0)
            else:
                legal_moves = list(board.legal_moves)
                move = random.choice(legal_moves) if legal_moves else None

            if move is None:
                break

            board.push(move)
            moves += 1

        result = board.result()  # '1-0', '0-1', '1/2-1/2'
        if result == "1-0":
            wins += 1
        elif result == "0-1":
            losses += 1
        else:
            draws += 1

    return wins, draws, losses


# ==========================================================
# 7. PROBE POSITIONS (OPTIONAL DEBUG / SANITY)
# ==========================================================
def probe_positions(agent: DQNAgent):
    """
    Print values for some hand-picked positions to see ordering.
    """
    # Start position
    start_board = chess.Board()
    v_start = agent.get_value(start_board)

    # White is totally winning (e.g., huge material)
    win_board = chess.Board("8/8/8/8/8/8/1K6/k7 w - - 0 1")
    v_win = agent.get_value(win_board)

    # Black to move in position where White is winning hard
    lose_board = chess.Board("8/8/8/8/8/8/k7/1K6 b - - 0 1")
    v_lose = agent.get_value(lose_board)

    print("Probe values:")
    print("  start:          ", v_start)
    print("  white big win:  ", v_win)
    print("  white big loss: ", v_lose)


# ==========================================================
# 8. TRAINING LOOP (GENERATOR) WITH METRICS
# ==========================================================
def train_dqn_generator(
    episodes: int = 200,
    capacity: int = 5000,
    eval_interval: int = 10,
    use_mlflow: bool = True,
    verbose: bool = False,
):
    """
    Generator that yields (board, episode, moves_count, info) after every move.

    info = {
        "q_val": current value estimate from current side's perspective,
        "winner": "White" / "Black" / "Draw" / None,
        "epsilon": current epsilon,
        "episode": current episode_idx,
    }
    """

    # ---- MLflow setup ----
    if use_mlflow:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("Chess_DeepQNetwork_RL")

    model = ChessNet()
    buffer = ReplayBuffer(capacity=capacity)
    agent = DQNAgent(model, buffer, gamma=0.99, lr=1e-3, target_update_steps=1000)

    # To allow code to run even without MLflow server
    mlflow_run_ctx = (
        mlflow.start_run(run_name=f"Training_Session_{episodes}_eps")
        if use_mlflow
        else None
    )

    try:
        if use_mlflow:
            mlflow.log_params(
                {
                    "episodes": episodes,
                    "algorithm": "DQN_ShapedReward",
                    "gamma": agent.gamma,
                    "lr": 1e-3,
                    "buffer_capacity": capacity,
                    "target_update_steps": agent.target_update_steps,
                }
            )

        for episode in range(1, episodes + 1):
            board = chess.Board()
            moves_count = 0
            episode_loss = 0.0
            total_reward = 0.0
            result_flag = 0.0  # from White's perspective: 1 win, -1 loss, 0 draw/unknown

            info = {
                "q_val": 0.0,
                "winner": None,
                "epsilon": 1.0,
                "episode": episode,
            }

            # Initial board (before any move)
            yield board, 0, 0, info

            while not board.is_game_over() and moves_count < 100:
                # Epsilon decay over episodes
                epsilon = max(0.1, 1.0 - (episode / (episodes * 0.8)))
                info["epsilon"] = epsilon

                # Perspective: current mover
                player_color = board.turn

                # Value estimate for current position
                info["q_val"] = agent.get_value(board)

                # Material BEFORE move
                material_before = get_material_value(board, player_color)

                # Choose move
                move = agent.choose_move(board, epsilon=epsilon)
                if move is None:
                    break

                prev_state = board_to_tensor(board).to(DEVICE)

                board.push(move)
                moves_count += 1

                # Yield for any visualization (after move)
                yield board, episode, moves_count, info

                # Material AFTER move (still viewing from same player's perspective)
                material_after = get_material_value(board, player_color)
                material_change = material_after - material_before

                # Reward shaping
                reward = 0.1 * material_change   # material bonus
                reward -= 0.005                  # small living penalty

                done = False
                if board.is_game_over():
                    done = True
                    result = board.result()  # '1-0', '0-1', '1/2-1/2'

                    # Game result from perspective of the player who JUST moved
                    if result == "1-0":
                        # White won
                        final_reward = 1.0 if player_color == chess.WHITE else -1.0
                        info["winner"] = "White"
                        result_flag = 1.0
                    elif result == "0-1":
                        # Black won
                        final_reward = 1.0 if player_color == chess.BLACK else -1.0
                        info["winner"] = "Black"
                        result_flag = -1.0
                    else:
                        final_reward = 0.0
                        info["winner"] = "Draw"
                        result_flag = 0.0

                    reward += final_reward

                total_reward += reward
                next_state = board_to_tensor(board).to(DEVICE)

                # Store transition
                buffer.push(prev_state, reward, next_state, done)

                # Optimize
                loss = agent.optimize(batch_size=32)
                episode_loss += loss

                if verbose:
                    print(
                        f"[Ep {episode} | Move {moves_count}] "
                        f"MatΔ: {material_change:+} "
                        f"R: {reward:+.3f} "
                        f"Q: {info['q_val']:+.3f} "
                        f"Loss: {loss:.4f}"
                    )

            avg_loss = episode_loss / max(1, moves_count)

            # ---- Logging ----
            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "loss": avg_loss,
                        "moves": moves_count,
                        "episode_total_reward": total_reward,
                        "episode_result_white_view": result_flag,
                    },
                    step=episode,
                )

            print(
                f"[EP {episode:3d}] Moves={moves_count:3d} "
                f"AvgLoss={avg_loss:.4f} "
                f"TotalR={total_reward:+.3f} "
                f"Result(White)={result_flag:+.0f}"
            )

            # ---- Periodic eval vs random ----
            if episode % eval_interval == 0:
                wins, draws, losses = play_vs_random(agent, n_games=5, max_moves=80)
                if use_mlflow:
                    mlflow.log_metrics(
                        {
                            "eval_wins_white_vs_random": wins,
                            "eval_draws_white_vs_random": draws,
                            "eval_losses_white_vs_random": losses,
                        },
                        step=episode,
                    )
                print(
                    f"[Eval @ Ep {episode}] "
                    f"W:{wins} D:{draws} L:{losses}"
                )

            # ---- Save model every few episodes ----
            if episode % 20 == 0 or episode == episodes:
                filename = f"chess_model_{episode}ep.pth"
                torch.save(model.state_dict(), filename)
                if use_mlflow:
                    try:
                        mlflow.log_artifact(filename, artifact_path="models")
                    except Exception:
                        pass

        # Final probe (optional)
        print("Final probe on some positions:")
        probe_positions(agent)

    finally:
        if mlflow_run_ctx is not None:
            mlflow_run_ctx.end()

    # Final yield: trained model
    yield model, -1, -1, {}



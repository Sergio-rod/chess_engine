#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import re
import glob
import dask.dataframe as dd
import torch
import chess
from pathlib import Path
from tqdm import tqdm
import gc
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split


# In[27]:


def generate_full_uci_move_vocabulary()-> tuple[dict,dict]:

    """
    Generates a set of all the posible movements in a chess board of 64 squares, 
    create two dictionaries which will represent the board move in uci format and their respective idx value and viceversa
    
    Returns
    -------
        uci_to_idx : dict
                     All the possible uci moves in a chess board, uci format as keys and idx as values
        idx_to_uci : dict 
                     All the possible uci moves in a chess board, uci format as keys and idx as values
    
    """
    move_set = set()
    
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
                
            from_rank = chess.square_rank(from_sq)
            to_rank = chess.square_rank(to_sq)
            from_file = chess.square_file(from_sq)
            to_file = chess.square_file(to_sq)
            rank_diff = abs(from_rank - to_rank)
            file_diff = abs(from_file - to_file)

            if file_diff==0 and rank_diff >= 1 :
                move_set.add(chess.Move(from_sq,to_sq).uci())
            if rank_diff ==0 and file_diff >= 1:
                move_set.add(chess.Move(from_sq,to_sq).uci())
            if rank_diff == file_diff:
                move_set.add(chess.Move(from_sq,to_sq).uci())
            if (rank_diff == 1 and file_diff == 2) or (rank_diff == 2 and file_diff == 1):
                move_set.add(chess.Move(from_sq,to_sq).uci())
            if (from_rank == 1 and to_rank == 0) or (from_rank == 6 and to_rank == 7):
                    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                        move_set.add(chess.Move(from_sq, to_sq, promotion=promo).uci())
            
            
    move_list = sorted(move_set)
    
    print(f"Total of valid movements generated: {len(move_list)}")
    print(f"Preview: {move_list[:10]}...")
    uci_to_idx = {uci: idx for idx, uci in enumerate(move_list)}
    idx_to_uci = {idx: uci for idx, uci in enumerate(move_list)}
    
    
    return uci_to_idx,idx_to_uci

    
def fen_to_tensor(fen:str) -> torch.Tensor:
    """
    Converts a FEN position into a torch tensor of shape (18,8,8),
    18 channels that will represent the possible states of a chess board of 8x8 positions.
    Channel 0-11: Positions for pieces PNBRQK(white pieces) or pnbrqk(black pieces)
    Channel 12: White kingside castling
    Channel 13: White queenside castling
    Channel 14: Black kingside castling
    Channel 15: Black queenside castling
    Channel 16: En passant target square
    Channel 17: Turm (1.0 for white or 0.0 for black)

    Parameters
    ----------
    fen : str
          The notation FEN to convert into tensor
    Returns
    -------
    board_tensor : torch.Tensor
                   The representation of FEN notation in 12 matrix of 8x8

    """

    board = chess.Board(fen)
    
    piece_to_index = {piece:idx for idx,piece in enumerate('PNBRQKpnbrqk')}

    #12 pieces + 4 castlings + 1 passant + 1 turn = 18 channels
    board_tensor = torch.zeros((18,8,8),dtype=torch.float32)

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = piece_to_index[piece.symbol()]
            row = 7 - (square // 8)
            col = square %8
            board_tensor[idx,row,col] = 1.0

    
    #Channels 12-15: Is Castling available 
    board_tensor[12, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    board_tensor[13, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    board_tensor[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    board_tensor[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    #Channel 16 en passant available
    
    if board.ep_square is not None:
        ep_row = 7 - (board.ep_square //8)
        ep_col = board.ep_square % 8
        board_tensor[16,ep_row,ep_col] = 1.0
    #Channel 17 Turn 
    board_tensor[17,:,:] = 1.0 if board.turn == chess.WHITE else 0.0
            
    return board_tensor


def get_legal_moves_vocab(fen:str) -> tuple[dict[str,int],dict[int,str]]:
    """
    Generates a set of legal posible moves for a given position 

    IMPORTANT ---> All the dict generated are LOCAL and could not match with the global dict --> generate_full_uci_move_vocabulary()

    Parameters
    ----------
        fen: FEN notation of the current position
    Returns
    -------
        uci_to_idx: Dict {uci_move : idx}
        idc_to_uci: Dict {idx : uci_move}
    """

    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    
    legal_moves_sorted = sorted(legal_moves, key=lambda m: m.uci())

    uci_to_idx = {move.uci():  idx for idx, move in enumerate(legal_moves_sorted)}
    idx_to_uci = {idx: move.uci() for idx,move in enumerate(legal_moves_sorted)}
    return uci_to_idx, idx_to_uci

 ## POSIBLEMENTE DESCARTADO, MEJORA ALTERNATIVA CON FUNCION  --> get_legal_moves_vocab enfoque "SPARSE"
# def get_legal_mask(board: chess.Board, uci_to_index: dict) -> torch.Tensor:
#     mask = torch.zeros(len(uci_to_index), dtype=torch.float32)
#     for move in board.legal_moves:
#         uci = move.uci()
#         if uci in uci_to_index:
#             mask[uci_to_index[uci]] = 1.0
#     return mask  # Shape: (n_moves,)

    


    # move_list = sorted(move_set)
    # uci_to_index = {uci: idx for idx, uci in enumerate(move_list)}
    # index_to_uci = {idx: uci for uci, idx in uci_to_index.items()}
    # return uci_to_index, index_to_uci
# Globales cargados una vez al inicio

def move_to_index(uci_move: str) -> int:
    return uci_to_index.get(uci_move, -1)  # -1 si no está

def index_to_move(idx: int) -> str:
    return index_to_uci.get(idx, "0000")  # dummy por si acaso

class ChessSequenceDataset(torch.utils.data.Dataset):


    def __init__(self,*,uci_to_idx=None,df=None,games=None):
        """
        Initializes the ChessSequenceDataset with either a dataframe of game moves or a list of game tensors.
    
        Parameters
        ----------
            uci_to_idx:      dict {uci_move : idx}
                             Dictionary mapping UCI (Universal Chess Interface) moves to a numerical value. The dictionary represents all possible move classes.
                             
            df:    pd.DataFrame
                             Pandas dataframe which will have a variety of columns, game_id,pyl,fen,and move_uci are relevant for this constructor.
                             
            games: list(list(Torch.tensor))
                             List of half moves represented as Torch.tensors which are the snapshots of a given FEN(Forsyth-Edwards Notation) nested in other list with full games.
                             Specifications:
                             n_games: number of games
                             n_moves_per_game: moves per game
                             dim: dimensions of each move's tensor 8*8*12
    
                             The total dimension: n_games*n_moves_per_game*dim
                    
        Notes
        -----
        Either 'df' or 'games' should be provided to create a dataset instance for training an LSTM model.
        
        """
        if games is not None:
            self.games = games
            self.uci_to_idx = uci_to_idx
        elif df is not None and uci_to_idx is not None:
            self.games = []
            self.uci_to_idx = uci_to_idx
            grouped = df.groupby('game_id')
    
            for game_id, group in grouped:
                group_sorted = group.sort_values(by='ply',ascending=True)
                sequence = []
    
                for _,row in group_sorted.iterrows():
                    fen = row['fen']
                    uci = row['move_uci']
    
                    move_idx = uci_to_idx.get(uci,-1)
                    if move_idx ==-1:
                        continue
                    try:
                        fen_tensor = fen_to_tensor(fen)
                    except Exception as e:
                        print(f'Error in FEN {fen} --->{e}')
                        continue
                    sequence.append((fen_tensor,move_idx,fen))
                if len(sequence)>0:
                    self.games.append(sequence)
    @classmethod
    def from_multiple_files(cls,file_list,uci_to_idx,stop_idx):
        """
        Initializes the ChessSequenceDataset with a file list of games previously processed, within a range of a given index

        Parameters
        ----------
            file_list: list
                list of paths of files which contain the processed games
            uci_to_idx: dict {uci_move: idx}
                Dictionary mapping UCI (Universal Chess Interface) moves to a numerical value. The dictionary represents all possible move classes.
            stop_idx: int
                Determine if the loading should stop in a certain number of processed dataset tho save resources.
                
        """
        
        all_games = []
        for _,file in enumerate(file_list):
            if _ > stop_idx:
                break
            games = torch.load(file)
            all_games.extend(games)

        return cls(games=all_games,uci_to_idx=uci_to_index)
            
    def __len__(self):
        return len(self.games)
    def __getitem__(self,idx):
        return self.games[idx]
        
def identity_collate(batch):
    return batch[0]  # simplemente devuelve la secuencia tal cual

class ChessLSTMPolicyNet(torch.nn.Module):
    def __init__(self,input_channels=18,lstm_hidden_size=128,lstm_layers=1,output_dim=2304):
        """
        Initialize the ChessLSTMPolicyNet model, inheriting class methods of module torch.nn.Module

        Parameters
        ----------
            input_channels: int(18)
                Channels that will represent the possible states of a chess board of 8x8 positions. fen_to_tensor()
            lstm_hidden_size: int(512)
                Quantity of LSTM neurons which will be the model trained on
            lstm_layers: int(1)
                Quantity of layers LSTM
            output_dim: int(4544)
                Quantity of possible move classes
        """
        

        
        super().__init__()

        
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels,64,kernel_size=3,padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,kernel_size=3,padding=1),
            torch.nn.ReLU()
        )
        self.flattened_size = 128*8*8
        self.lstm = torch.nn.LSTM(
            input_size= self.flattened_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True
        )


        self.fc = torch.nn.Linear(lstm_hidden_size,output_dim)


    def forward(self,x):
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)           # (B*T, C, 8, 8)
            x = self.conv(x)                     # (B*T, 128, 8, 8)
            x = x.view(B, T, -1)                 # (B, T, 8192)

            lstm_out, _ = self.lstm(x)           # (B, T, hidden)
            logits = self.fc(lstm_out)           # (B, T, output_dim)
            return logits

        
def train_lstm(model,uci_to_idx_global, train_dataloader,val_dataloader, criterion, optimizer,device, epochs=5,start_epoch=0, checkpoint_path='checkpoint.pth'):
    """
    Function to train the model for a given dataset, saving each epoch result to segregate the traning into smaller blocks of processing

    Parameters
    ----------
        model: ChessLSTMPolicyNet
            Architecture of the model that define the flow of how the dataloader will be processed for machine learning.
        dataloader: torch.utils.data.DataLoader
            Iterator of lenght of n_games/batch_size, normally batch_size will be equal to 1, considering that is needed process just complete sequences of a game 
        criterion: torch.nn.CrossEntropyLoss()
            Loss function to calculate the loss 
        optimizer: torch.optim.Adam()
            Opimizer which will help us to optimze the weights in the neural network based on the results of the loss function
        device:
            Device to perform the training, usually a GPU
        epochs:
            Number of epochs to iterate the training
        start_epoch:
            Index from where will be start the training
        checkpoint_path: 
            File with the saved training epochs """
    history = {
        'epoch': [],
        'train_acc':[],
        'train_loss':[],
        'val_acc':[],
        'val_loss':[],
        'lr': []
    }

    for epoch in range(start_epoch,start_epoch+epochs):
        total_train_loss = 0
        total_train_correct = 0
        total_train_moves = 0
        model.train()

        for game in train_dataloader:
            # batch: list of 1 element (game of (fen_tensor, move_idx))
            if not all(isinstance(x,tuple) and isinstance(x[0],torch.Tensor) for x in game):
                print('Invalid game detected and omitted in train train_dataloader...')
                continue
            try: 
                # Dataset attributes
                inputs = torch.stack([x[0] for x in game])  # Representation of FEN in torch tensor (T, C, 8, 8)
                targets = torch.tensor([x[1] for x in game],dtype=torch.long)  # (T,)
                fens = [x[2] for x in game]
                
            # Reshape to (B, T, C, 8, 8)
                inputs = inputs.unsqueeze(0).to(device)
                targets = targets.unsqueeze(0).to(device)

                optimizer.zero_grad()
                outputs = model(inputs)  # (B, T, output_dim)

            # Aplanar para CrossEntropy
                logits = outputs.view(-1, outputs.size(-1))     # (Channels = 18, 8, 8)
                masked_logits = mask_logits(uci_to_idx_global,fens,logits)
                target_flat = targets.view(-1)                  # (T,)

                loss = criterion(masked_logits, target_flat)
                loss.backward()
                optimizer.step()
    
                total_train_loss += loss.item()
                preds =  torch.argmax(masked_logits, dim=1)#torch.argmax(logits, dim=1)
                total_train_correct += (preds == target_flat).sum().item()
                total_train_moves += target_flat.size(0)
            except Exception as e:
                print(f'Something went wrong: {type(e).__name__}: {e}')
                continue
        train_acc = total_train_correct / total_train_moves
        train_loss = total_train_loss/len(train_dataloader)

    ##add validation
        model.eval()
    
        total_val_loss = 0
        total_val_correct = 0
        total_val_moves = 0
        with torch.no_grad():
            for game in val_dataloader:
                if not all(isinstance(x,tuple) and isinstance(x[0],torch.Tensor) for x in game):
                    print('Invalid game detected and omitted in val_dataloader...')
                    continue
                try: 
                        
                    inputs = torch.stack([x[0] for x in game])  # (Channels = 18, 8, 8)
                    targets = torch.tensor([x[1] for x in game],dtype=torch.long)  # idx_move 
                    fens = [x[2] for x in game]
        
                    # Reshape to (B, T, C, 8, 8)
                    inputs = inputs.unsqueeze(0).to(device)
                    targets = targets.unsqueeze(0).to(device)
                        
                    outputs = model(inputs)  # (B, T, output_dim)
        
                     
        
                    # Aplanar para CrossEntropy
                    logits = outputs.view(-1, outputs.size(-1))     # (T, output_dim)
                    masked_logits=mask_logits(uci_to_idx_global,fens,logits)
                    target_flat = targets.view(-1)                  # (T,)
                    loss = criterion(masked_logits,target_flat)
                    total_val_loss += loss.item()
                    preds = torch.argmax(masked_logits, dim=1)
                    total_val_correct += (preds == target_flat).sum().item()
                    total_val_moves += target_flat.size(0)
                                
                except Exception as e:
                    print(f'Something went wrong during validation: {type(e).__name__}: {e}')
                
            val_acc = total_val_correct/total_val_moves
            val_loss = total_val_loss/len(val_dataloader)

        # --- LOGGING ---
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")  # CORRECCIÓN: Mensaje completo
        history['epoch'].append(epoch + 1)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Save checkpoint with both train and validation metrics
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
        }, checkpoint_path)
    return history
            

        
def mask_logits(uci_to_idx_global, fens, logits):
    masked_logits = []
    missing_moves = set()
    
    for t, fen_str in enumerate(fens):
        board = chess.Board(fen_str)
        mask = torch.full((logits.size(1),), float('-inf'), device=logits.device)
        
        for move in board.legal_moves:
            uci = move.uci()
            if uci in uci_to_idx_global:
                mask[uci_to_idx_global[uci]] = 0
            else:
                missing_moves.add(uci)  # Para debugging
        
        if missing_moves:
            print(f"⚠️ Movimientos faltantes en el vocabulario: {missing_moves}")
            
        masked_logits.append(logits[t] + mask)
    
    return torch.stack(masked_logits, dim=0)       

# def mask_logits(uci_to_idx_global,fens,logits):
#     masked_logits = []
#     for t, fen_str in enumerate(fens):
#         board = chess.Board(fen_str)
#         legal_idxs = [uci_to_idx_global[m.uci()] for m in board.legal_moves if m.uci() in uci_to_idx_global]
#         mask = torch.full((logits.size(1),),fill_value=float('-inf'),device=logits.device)
#         mask[legal_idxs] = 0
#         masked_logits.append(logits[t] + mask)
#     masked_logits = torch.stack(masked_logits,dim=0)
#     return masked_logits


# In[2]:


def save_dataset_chunks(df, uci_to_idx,chunk_size=50_000, output_prefix='chunk_sequence'):
    game_ids = df['game_id'].unique()
    total_games = len(game_ids)
    num_chunks = (total_games + chunk_size -1) // chunk_size

    for i in range(num_chunks):
        save_path = Path.cwd().parent /'data'/'torch_datasets'/f'{output_prefix}_{i:04d}.pt'
        save_path.parent.mkdir(parents=True,exist_ok=True)
        print(f'Veryfing if chunk number: {i:04d} exists...')

        if os.path.exists(save_path):
            print(f'Chunk {i:04d} exist, skipping...')
            continue
        else:
            start = i*chunk_size
            end = min(start + chunk_size,total_games)
            chunks_ids = game_ids[start:end]
            chunk_df = df.loc[df['game_id'].isin(chunks_ids)]
    
            print(f'Processing chunk number {i}/{num_chunks} with {len(chunks_ids)} games...')
            dataset = ChessSequenceDataset(chunk_df,uci_to_idx)
    
            torch.save(dataset.games,save_path)
            print(f'[💾] Saved: {save_path} with {len(dataset)} valid games')
    

        
    

# torch_datasets = glob.glob(str(Path.cwd().parent / 'data'/'torch_datasets'/'*.pt'))

# parquet_datasets = glob.glob(str(Path.cwd().parent / 'data' / 'processed' /'*.parquet'))

# df_sample = pd.read_parquet(parquet_datasets[0])




# In[8]:


# moves_count = df_sample['game_id'].value_counts()
# moves_count


# In[20]:



# In[1]:


# import plotly.graph_objects as go

# moves_count = df_sample['game_id'].value_counts()

# fig = go.Figure(data=[
#     go.Histogram(
#         x=moves_count.values,
#         nbinsx=50  # puedes ajustar el número de bins (ej. 50, 100, etc.)
#     )
# ])

# fig.update_layout(
#     title="Distribución de jugadas por partida",
#     xaxis_title="Número de jugadas",
#     yaxis_title="Número de partidas",
#     bargap=0.05
# )

# fig.show()


# In[28]:


# uci_to_idx,idx_to_uci = generate_full_uci_move_vocabulary()

# unique_games = df_sample['game_id'].unique()

# sample_games =np.random.choice(unique_games,size=15*10**3,replace=False)


# X_train,X_val= train_test_split(sample_games,test_size=.2,random_state=42)
# df_sample_train = df_sample.loc[df_sample['game_id'].isin(X_train)].copy()
# df_sample_val = df_sample.loc[df_sample['game_id'].isin(X_val)].copy()

# if uci_to_idx:
#     train_dataset = ChessSequenceDataset(uci_to_idx=uci_to_idx,df=df_sample_train)
#     val_dataset = ChessSequenceDataset(uci_to_idx=uci_to_idx,df=df_sample_val)

    

# train_dataloader =torch.utils.data.DataLoader(train_dataset,batch_size=1,collate_fn=identity_collate)
# val_dataloader =torch.utils.data.DataLoader(val_dataset,batch_size=1,collate_fn=identity_collate)


# In[ ]:


# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ChessLSTMPolicyNet().to(device)

# history = train_lstm(
#     model=model,
#     uci_to_idx_global = uci_to_idx,
#     train_dataloader=train_dataloader,
#     val_dataloader= val_dataloader,
#     criterion=torch.nn.CrossEntropyLoss(),
#     optimizer=torch.optim.Adam(model.parameters(),lr=1e-3),
#     device=device
# )


# In[ ]:

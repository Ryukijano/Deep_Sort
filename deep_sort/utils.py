from pyexpat import model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def ask_user():
    print("Write your array as a list with arbitary positive numbers:")
    array = input("Input q if you want to quit \n")
    return array

def sort_array(encoder, decoder, device, arr=None):
    """
    A simple example use of the model
    Input: encoder.nn.Module
           decoder nn.Module
           device
           array to sort(optional)
           
    """
    
    if arr is None:
        arr = ask_user()
        
    with torch.no_grad():
        while arr != 'q':
            #Avoding numerical errors by rounding to max_len
            arr = eval(arr)
            lengths = [
                len(str(elem).split(".")[1]) if len(str(elem).split("."))>1 else 9
                for elem in arr
            ]
            
            max_len = max(lengths)
            source = torch.tensor(arr, dtype=torch.float).to(device).unsqueeze(1)
            batch_size = source.shape[1]
            target_len = source.shape[0] + 1
            
            outputs = torch.zeros(target_len, batch_size, target_len - 1 ).to(device)
            encoder_states, hidden, cell  = encoder(source)
            
            #First input will be <SOS> token
            x = torch.tensor([[1]], dtype=torch.float).to(device)
            predictions = torch.zeros(target_len, batch_size, dtype=torch.float).to(device)
            
            for t in range(1, target_len):
                #At every time use encoder_states and hidden states to predict next token
                attention, energy, hidden, cell = decoder(x, encoder_states, hidden, cell)
                
                #store predictions for each time step
                outputs[t] = energy.permute(1, 0)
                
                #Get the best word for each batch(index in the vocabulary)
                best_guess = attention.argmax(0)
                predictions[t] = best_guess.item()
                x = torch.tensor([[best_guess.item()]], dtype=torch.float).to(device)
                
            output =[
                    round(source[predictions[1:,i].item()].item(), max_len) 
                    for i in range(source.shape[0])
                    ]
            
            print(f"Here's the result: {output}")
            arr = ask_user()
            

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)
    

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    steps = checkpoint['steps']
    return steps 
    
                
            
            
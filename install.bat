python -m venv venv
call .\venv\Scripts\activate

git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
git pull
pip install -U -r requirements.txt
cd ..

pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.1.post1-py3-none-win_amd64.whl
pip install https://github.com/bdashore3/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

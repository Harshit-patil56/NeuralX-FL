shared_data/ — Data directory for REAL NETWORK mode only
==========================================================

PURPOSE
-------
This folder holds the data files you hand to your friends (clients).
It is SEPARATE from ../data/ which is used only for Ray simulation.

This separation clearly shows:
  - data/          → simulation only, stays on your machine
  - shared_data/   → what you physically gave to each friend's laptop


WHAT GOES HERE
--------------
  pathmnist.npz    — PathMNIST dataset file

Copy it from ../data/pathmnist.npz:
  Windows:  copy ..\data\pathmnist.npz .\shared_data\pathmnist.npz
  Linux/Mac: cp ../data/pathmnist.npz ./shared_data/pathmnist.npz

Then hand this entire project folder (with shared_data/pathmnist.npz
already inside) to your friends via USB or file share.


HOW DATA STAYS PRIVATE
----------------------
The pathmnist.npz contains pathology images.
When a friend runs  `python run_client.py --partition-id 0`:

  1. Their laptop loads images from THIS folder locally.
  2. Trains the CNN model locally (never uploads images).
  3. Sends only the updated MODEL WEIGHTS to your server.
  4. Your server never sees raw images — only floating-point arrays.

Proof:
  - Open Wireshark on the friend's laptop.
  - Capture filter: tcp.port == 8080
  - Inspect any packet → you see gRPC protobuf with float32 arrays.
  - You do NOT see image data (28x28 RGB = 2352 bytes per image).
  - The actual packets are ~200-400 KB (model weights), not ~200 MB (images).


DO NOT
------
  - Do not put this folder's contents into ../data/
  - Do not commit pathmnist.npz to git (add to .gitignore)

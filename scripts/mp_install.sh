pip install nats_bench
git clone --recurse-submodules https://github.com/D-X-Y/AutoDL-Projects.git XAutoDL
cd XAutoDL/
pip install .
pip install progress
cd /content/drive/MyDrive/Expirement/MP-NAS-Bench201/
python generate_data.py --index 131-140 --device 0 --total 1000 --cell_type cell_group_op_random
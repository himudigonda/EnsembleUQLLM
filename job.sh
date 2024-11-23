#! /bin/bash
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 0-1
#SBATCH --gres=gpu:a100:1
#SBATCH -p general
#SBATCH -q public
#SBATCH --job-name=LLM-UQ
#SBATCH --output=./LLM-UQ-%j.out
#SBATCH --error=./LLM-UQ-%j.err
#SBATCH --mem=80G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hmudigon@asu.edu

# Function to echo the current time
echo_time() {
	echo "Timestamp: [$(/bin/date '+%Y-%m-%d %H:%M:%S')]......................................................$1"
}

echo "===== himudigonda ====="
echo ""
echo ""

echo_time "[1/5] Loading CUDA 12.4 and Mamba"
module load cuda-12.4.1-gcc-12.1.0
echo_time "[+] Done"
echo ""

echo_time "[2/5] Activating virtual environment"
source activate qullm
echo_time "[+] Done"
echo ""

echo_time "[3/5] Changing working directory"
cd /scratch/hmudigon/codes/EnsembleUQLLM
echo_time "[+] Done"
echo ""

echo_time "[4/5] Initiating training..."
sh train.sh
echo_time "[+] Done"
echo ""

echo_time "[5/5] Initiating inference..."
sh infer.sh
echo_time "[+] Done"
echo ""
echo ""

echo_time "[+] Execution completed successfully!"
echo ""
echo "===== himudigonda ====="

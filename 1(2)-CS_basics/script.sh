#!/bin/bash

# anaconda(또는 miniconda)가 존재하지 않을 경우 설치해주세요!
if ! command -v conda &> /dev/null; then
    echo "[INFO] miniconda 설치 시작"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
else
    echo "[INFO] conda 이미 설치됨"
fi

# Conda 환셩 생성 및 활성화
conda init bash
source ~/.bashrc
conda create -n myenv python=3.10 -y
conda activate myenv

## 건드리지 마세요! ##
python_env=$(python -c "import sys; print(sys.prefix)")
if [[ "$python_env" == *"/envs/myenv"* ]]; then
    echo "[INFO] 가상환경 활성화: 성공"
else
    echo "[INFO] 가상환경 활성화: 실패"
    exit 1 
fi

# 필요한 패키지 설치
pip install mypy

# Submission 폴더 파일 실행
cd submission || { echo "[INFO] submission 디렉토리로 이동 실패"; exit 1; }

for file in *.py; do
    problem_id=$(echo "$file" | cut -d'.' -f1)
    echo "[INFO] 실행 중: $file"
    input_file="../input/${problem_id}_input"
    output_file="../output/${problem_id}_output"
    
    if [[ -f "$input_file" ]]; then
        python "$file" < "$input_file" > "$output_file"
        echo "[INFO] 완료: $file"
    else
        echo "[WARN] 입력 파일 없음: $input_file"
    fi
done

# mypy 테스트 실행 및 mypy_log.txt 저장
cd ..
mypy submission > test_log.txt

# conda.yml 파일 생성
conda env export > conda.yml

# 가상환경 비활성화
conda deactivate

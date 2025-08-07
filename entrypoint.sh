# Create Data dirctories
mkdir Data

mkdir Data/2017
mkdir Data/2017/chunks

mkdir Data/2018
mkdir Data/2018/chunks

mkdir Data/2019
mkdir Data/2019/chunks

mkdir Data/2020
mkdir Data/2020/chunks

mkdir Data/2021
mkdir Data/2021/chunks

mkdir Data/2022
mkdir Data/2022/chunks

mkdir Data/2023
mkdir Data/2023/chunks

mkdir Data/2024
mkdir Data/2024/chunks


# Run script to get the data
python retrieve_data.py
python split_chunks.py
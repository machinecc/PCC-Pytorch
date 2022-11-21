call D:\\anaconda3\\Scripts\\activate.bat
call conda activate pytorch
call python main2.py -bw 2
call python main2.py -bw 5
call python main2.py -bw 10
call python main2.py -bw 15
call python main2.py -bw 20
call python main2.py --reward latency -bw 2
call python main2.py --reward latency -bw 5
call python main2.py --reward latency -bw 10
call python main2.py --reward latency -bw 15
call python main2.py --reward latency -bw 20
REM python main2.py --analysis True
pause

import sys,os
sys.path.append(os.path.join(os.getcwd(),'../'))
from concurrent import futures
import time
import grpc
from example import PaddleXserver_pb2_grpc
from Server_core import PaddleXserver

def server():
    #启动 rpc 服务
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    PaddleXserver_pb2_grpc.add_PaddleXserverServicer_to_server(PaddleXserver(),server)
    server.add_insecure_port('0.0.0.0:50000')
    server.start()
    print('server starting')
    try:
        while True:
            time.sleep(60*60*24) #一天，按秒算
    except KeyboardInterrupt:
        server.stop(0)
        
if __name__ == '__main__':
    server()
from socket import *
from time import ctime
import threading

config = 1
old_config = 1
machines = []
epoch_in_machines = []
epoch_in_progress = 0
epoch_in_progress_changed = True
config_updated = []
next_epoch = [] # 每个进程是否收到了进入下一个进程的信号

def read(socket_fuw, epoch_lock): # 同步不同的进程
    global epoch_in_machines
    global machines
    while True:
        recv_data = socket_fuwu.recv(1024)
        epoch_lock.acquire()
        if recv_data:
            recv_data = recv_data.decode('utf-8')
            print('rank %d finishes epoch %d' % (machines.index(socket_fuwu), 
            	int(recv_data.split(":")[1])))
            print('rank %d finishes epoch %d' % (int(recv_data.split(":")[0]), 
            	int(recv_data.split(":")[1])))
            
            epoch_in_machines[int(recv_data.split(":")[0])] = int(recv_data.split(":")[1])
        epoch_lock.release()


def write(socket_fuwu, epoch_lock, config_lock): #改变config
    global config
    global old_config
    global epoch_in_machines
    global epoch_in_progress
    global machines
    global epoch_in_progress_changed
    global config_updated
    global next_epoch
    while True:
        config_lock.acquire()
        if config != old_config:
            if config_updated[machines.index(socket_fuwu)] == 0:
                socket_fuwu.send(("config:" + str(config)).encode('utf-8'))
                config_updated[machines.index(socket_fuwu)] = 1
            config_updated_count = 0
            for each in config_updated:
                if each != 0:
                    config_updated_count += 1
            if config_updated_count == len(machines):
                old_config = config
        config_lock.release()


        epoch_lock.acquire()
        flag = 1 # 是否允许所有进程进入下一个epoch
        for i in range(len(epoch_in_machines)): # 判断是否每一个进程都完成了当前epoch
            if epoch_in_machines[i] != epoch_in_progress:
                flag = 0
                break
        if flag == 1:
            # 给所有进程发信号
            socket_fuwu.send(str(1).encode('utf-8'))
            next_epoch[machines.index(socket_fuwu)] = 1
            print("rank %d ready for next epoch: %d" % (machines.index(socket_fuwu), epoch_in_progress))
            all_in_next_epoch = True
            for each in next_epoch:
                if each != 1:
                    all_in_next_epoch = False
                    break
            if all_in_next_epoch:
                epoch_in_progress += 1
                for i in range(len(next_epoch)):
                    next_epoch[i] = 0
        epoch_lock.release()

def input_data(config_lock):
    global config
    global old_config
    global config_updated
    while True:
        data=input('>')
        config_lock.acquire()
        config = int(data)#.encode('utf-8')
        print(config, old_config, config==old_config)
        for i in range(len(config_updated)):
            config_updated[i] = 0
        config_lock.release()

tcp_socket_host = socket(AF_INET,SOCK_STREAM)

# 服务器端口回收操作（释放端口）
tcp_socket_host.setsockopt(SOL_SOCKET, SO_REUSEADDR, True)

# 2绑定端口
tcp_socket_host.bind(('',8080))

# 3监听  变为被动套接字
tcp_socket_host.listen(128)    #128可以监听的最大数量，最大链接数

epoch_lock = threading.Lock()
config_lock = threading.Lock()

t = threading.Thread(target=input_data, args=(config_lock,))
t.start()

# 4等待客户端连接
while True:
    socket_fuwu,addr_client=tcp_socket_host.accept()  #accept(new_socket,addr)
    print(socket_fuwu)
    print(addr_client)
    machines.append(socket_fuwu)
    epoch_in_machines.append(None)
    next_epoch.append(1)
    config_updated.append(0)

    t1=threading.Thread(target=read,args=(socket_fuwu, epoch_lock)) # 保持各GPU间训练过程同步
    t1.start()

    # 给所有进程发信号以同步这些进程
    t2=threading.Thread(target=write,args=(socket_fuwu, epoch_lock, config_lock)) # the thread to send new config to machines
    t2.start()


#6服务套接字关闭
#socket_fuwu.close()    #服务器一般不关闭   此时服务端口因为需要一直执行所以也不能关闭


# Author:Han
# @Time : 2020/1/12 13:44
# -*- coding: utf-8 -*-
import socket
import os

def send_str():
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    edge_ip = '172.19.5.75'  # �ߵ�ip
    edge_port = 2000
    edge_addr = (edge_ip, edge_port)
    print('�������ӱ�Ե������')
    tcp.connect(edge_addr)
    print('���ӳɹ�')
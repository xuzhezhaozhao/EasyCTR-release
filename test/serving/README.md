
需要先编译出 libtensorflow_cc.so, 并且需要将自定义 op 打包进去

$ git clone https://github.com/xuzhezhaozhao/tensorflow
$ git checkout EasyCTR
$ ./configure
$ ./build_so.sh
$ ./make_dist.sh
$ sudo cp -r dist/include/tensorflow /usr/local/include
$ sudo cp dist/lib/libtensorflow_cc.so.1 /usr/local/lib/
$ sudo ln -s /usr/local/lib/libtensorflow_cc.so.1 /usr/local/lib/libtensorflow_cc.so

编译时还需要使用 --whole-archive 参数

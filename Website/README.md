# 项目的网站目录
## 网站环境
1. Python 3.6
2. Django 3.0
## 安装相关依赖库

`pip install -r requirements.txt`

## 开发

hurricane618

## 一些环境问题

### mysqlclient安装失败

检查自己是否安装mysql或者mysql-client，这个库需要mysql相关程序支持

Mac：`brew install mysql` 或者 `brew install mysql-client`

添加mysql的bin目录进PATH中

`export PATH="/usr/local/mysql/bin:${PATH}"`

再次安装即可解决问题

### 加载动态库问题

在Mac下加载mysql的动态库出现错误，是因为路径找不到的原因，我们给`/usr/local/lib`链接一个动态库即可

`sudo ln -s /usr/local/mysql/lib/libmysqlclient.21.dylib /usr/local/lib/libmysqlclient.21.dylib`

这里的动态库换成自己缺失的


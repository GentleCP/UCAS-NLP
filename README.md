# UCAS-NLP
UCAS自然语言处理的编程大作业



# 后台和模型之间交互

django/flask传递待分类的文本给模型，然后模型即时或异步返回分类结果（根据实际需要运行的时间确定）
```
text = "据可靠消息，今年信工所的学生将会安排在双人间，宿舍面积共计7平方米。同学实测后发现包括飘窗在内面积仍然不足7平方米。"
def get_result(text: str, **args) -> dict:
	"""
	模型预留一个函数，供网站后端调用，
	args: 某些必要的参数，根据模型的需要进一步确定
	text: 待分类的文本
	"""


	result = {
	"status": "success",
	"result": {"新闻": 0.9, "军事": 0.01}
	}
	return result 
```

网站的前后端数据传递由网站开发人员自行设计

前端根据后端返回的结果做可视化

建议d3或者echarts 网页加点图表显得炫酷 


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



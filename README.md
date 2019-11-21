# UCAS-NLP
UCAS自然语言处理的编程大作业



# 后台和模型之间交互

django/flask传递待分类的文本给模型
```
{
	"text": "据可靠消息，今年信工所的学生将会安排在人均3.5平方米的宿舍，中国科学院大学后勤管理处目前尚未发表回应"
}
```

模型返回分类结果
```
{
	"status": "success",
	"result": {"新闻": 0.9, "军事": 0.01}
}
```


网站的前后端数据传递由网站开发人员自行设计

前端根据后端返回的结果做可视化

建议d3或者echarts 网页加点图表显得炫酷 

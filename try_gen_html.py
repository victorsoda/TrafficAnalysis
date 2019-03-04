#coding:utf-8
_author_ = "LiaoPan"
_time_  = "2016.6.16"
_myblog_ = "http://blog.csdn.net/reallocing1?viewmode=contents"
import webbrowser

GEN_HTML = "demo_1.html"  #命名生成的html

str_1 = "1: new contents need to be added."
str_2 = "2: new contents need to be added."

f = open(GEN_HTML,'w')
message = """
<html>
<head></head>
<body>
<p>Hello,World!</p>
<p>Add webbrowser function</p>
<p>%s</p>
<p>%s</p>
</body>
</html>"""%(str_1,str_2)

f.write(message)
f.close()

webbrowser.open(GEN_HTML,new = 1)

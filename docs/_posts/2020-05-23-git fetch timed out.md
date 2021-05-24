---
author: rym
catagory: blog
---
不使用http方式,换成ssh方式。

设置ssh密钥。

<a href="https://www.cnblogs.com/yehui-mmd/p/5962254.html">github-如何设置SSH Key</a>

查看现有的remote:

````sh
git remote -v
````

然后更改remote
````
git remote rm origin
git remote add origin git@github.com:username/reponame.git
git push -u origin master
````

# rymoon.github.io
<https://rymoon.github.io>

## 如何上传文档
* 准备markdown格式的文件
* 该名为yyyy-mm-dd-title.md，比如2020-05-24-用户须知.md  ，title可以是任何无空格字串。
* 在文件开头加入以下行：
  ````md
  ---
  author: short_name
  ---
  ````
  其中short_name如下替换，表示作者缩写：
  |作者|缩写|
  |----|----|
  |张胜|vu|
  |张佳毅|vjy|
  |任雨濛|rym|
* 在markdown文件中,<!-- more -->之前的内容被视为摘要。这个语句是必须有的。
* 放到`rymoon.github.io/docs/_posts/`
* 图片放到`rymoon.github.io/docs/assets/image/post_name/`下，其中 post_name替换成markdown文档的名字。
  在markdown中通过`{{site.url}}/assets/image/post_name/xxx.png`进行引用，比如
  ````markdown
  ![xxx]({{site.url}}/assets/image/post_name/xxx.png)
  ````
* 使用latex时请注意：
  * $$ $$ 中的内容要包在环境里。比如：
  
    `$$ \lambda = 3 $$`
    
    是不能正常显示的。而
    
    ````latex
    $$ \begin{aligned}
    \lambda & =3 
    \end(aligned}$$
    ````
    
    才行。
    


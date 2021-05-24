---
layout: default
title: Home
---

![main_image]({{site.url}}/assets/image/88985676_p0_master1200.jpg)

欢迎来到 Rymoon的小站！

<a href="/blog.html">浏览全部博文</a>
<h1>最新文章</h1>
<ul>
  {% for post in site.posts limit:2 %}
    <li>
      <h2><a href="{{ post.url }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt | strip_html }}</p>
    </li>
  {% endfor %}
</ul>

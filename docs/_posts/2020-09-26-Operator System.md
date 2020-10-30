---
author: rym
catagory: blog
---
# 现代操作系统

> 本文节选自《现代操作系统，第四版/Modern Operating System, 4th ed》，1.1-1.5节。
> 本文只保留主干框架，详情阅读原文。

<!-- more -->

## 基本介绍

现代操作系统由一个或者多个处理器、主存、磁盘、打印机、键盘 、鼠标、显示器、网络接口以及各种其他输入输出设备组成。计算机安装了一个复杂的系统，为用户程序提供一个更好 、更简单、更清晰的计算机模型，并管理刚才提到的所有设备。

用户与之交互的程序，基于文本的通常称为shell，而基于图标的则称为图形用户界面(Graphical User Interface, GUI)，它们实际上并不是操作系统的一部分，尽管这些程序使用操作系统来完成工作。

操作系统有两个基本上独立的任务，即为应用程序提供一个资源集的清晰抽象，并管理这些硬件资源。

![操作系统所处的位置]({{site.url}}/assets/image/2020-09-26-OS/image1-1.png "操作系统所处的位置")

## 作为扩展机器的操作系统

在机器语言一级上，多数计算机的体系结构（指令集、存储组织、I/O和总线结构）是很原始的，而且编程是很困难的，尤其对输入/输出操作而言。为了更细致地考察这一点，我们以大多数电脑使用的更现代非SATA(Serial ATA)硬盘为例。曾有一本描述早期硬盘接口（程序员为了使用硬盘而需要了解的东西）的书（Anderson,2007）它的页数超过450页。自2007年起，接口又被修改个很多次，因而比当时更加复杂。显然，没有任何理智的程序员想要在硬件层面上和硬盘打交道。相反，他们使用一些叫作硬盘驱动（disk driver）的软件和硬件交互。这类软件提供了读写硬盘块的接口，而不同深入细节。操作系统包含花很多用于控制输入输出设备的驱动。

但就算是在这个层面上，对于大多数应用来说还是太底层了。因此，所有操作系统都提供使用硬盘的又一层抽象：文件。使用该抽象，程序能创建、读写文件，而不用处理硬件实际工作中那些恼人的细节。

![操作系统将丑陋的硬件转变成为美丽的抽象]({{site.url}}/assets/image/2020-09-26-OS/image1-2.png "操作系统将丑陋的硬件转变成为美丽的抽象")

需要指出的是，操作系统的实际客户是应用程序（当然是通过应用程序员）。它们直接与操作系统及其抽象打交道。相反，最终用户与用户接口所提供的抽象打交道，或者是命令行shell或者是图形接口。而用户接口的抽象可以与操作系统提供的抽象类似，但也不总是这样。为了更清晰地说明这一点，请读者考虑普道的Windows桌面以及面向行的命令提示符。两者都是运行在Windows操作系统上的程序，并使用了Windows提供的抽象，但是它们提供了非常不同的用户接口。类似地，运行Gnome或者KDE的Linux用户与直接在xWindow系统（面向文本）顶部工作的Linux用户看到的是非常不同的界面，但是在这两种情形中，操作系统下面的抽象是相同的。

## 作为资源管理的操作系统

把操作系统看作向应用程序提供基本抽象的溉念，是一种自顶向下的观点。按照另一种自底向上的观点，操作系统则用来管理一个复杂系统的各个部分。现代计算机包含处理器、存储器、时钟、磁盘、鼠标、网络接口、打印机以及许多其他设备。从这个角度看，操作系统的任务是在相互竞争的程序之间有序地控制对处理器、存储器以及其他]/O接口设备的分配。

现代操作系统允许同时在内存中运行多道程序。假设在一台计算机上运行的三个程序试图同时在同一台打印机上输出计算结果，那么开始的几行可能是程序1的輪出，接着几行是程序2的输出，然后又是程序3的输出等，最终结果将是一团槽。采用将打印结果送到磁盘上缓冲区的方法，操作系统可以把潜在的混乱有序化。在一个程序结束后，操作系统可以将暂存在磁盘上的文件送到打印机输出，同时其他程序可以继续产生更多的输出结果，很明显，这些程序的输出还没有真正送至打印机。

当一个计算机（或网络）有多个用户时，管理和保护存储器、I/O设备以及其他资源的需求变得强烈起来，因为用户间可能会互相干扰。另外，用户通常不仅共享硬件，还要共享信息（文件、数据库等）。简而言之，操作系统的这种观点认为，操作系统的主要任务是记录哪个程序在使用什么资源，对资源请求进行分配，评估使用代价，并且为不同的程序和用户调解互相冲突的资源请求。

资源管理包括用以下两种不同方式实现多路复用（共享）资源：在时间上复用和在空间上复用。当一种资源在时间上复用时，不同的程序或用户轮流使用它·先是第一个获得资源的使用，然后下一个，以此类推。例如，若在系统中只有一个CPU，而多个程序需要在该CPU上运行，操作系统则首先把该CPU分配给某个程序，在它运行了足够长的时间之后，另一个程序得到CPU，然后是下一个，如此进行下去，最终，轮到第一个程序再次运行。至于资源是如何实现时间复用的一一谁应该是下一个以及运行多长时间等一一则是操作系统的任务。还有一个有关时间复用的例子是打印机的共享。当多个打印作业
在一台打印机上排队等待打印时，必須决定将轮到打印的是哪个作业。

另一类复用是空间复用。每个客户都得到资源的一部分，从而取代了客户排队。例如，通常在若干运行程序之间分割内存，这样每一个运行程序都可同时人驻内存（例如，为了轮流使用CPU）。假设有足够的内存可以存放多个程序，那么在内存中同时存放若干个程序的效率，比把所有内存都分给一个程序的效率要高得多，特别是，如果一个程序只需要整个内存的一小部分，结果更是这样心当然，如此的做法会引起公平、保护等问题，这有赖于操作系统解决它们。有关空间复用的其他资源还有磁盘。在许多系统中，一个磁盘同时为许多用户保存文件。分配磁盘空间并记录谁正在使用哪个磁盘块，是操作系统的典型任务。

## 操作系统的历史

### 第一代(1945-1955)：真空管和穿孔卡片

### 第二代（1955-1965）：晶体管和批处理系统

### 第三代(1965-1980)：集成电路和多道程序设计

### 第四代(1980-)：个人计算机

### 第五代（1990-）：移动计算机

## 计算机硬件简介

从概念上讲，一台简单的个人计算机可以抽象为类似于图1．6中的模型。CPU、内存以及I/O设备都由一条系统总线连接起来并通过总线与其他设备通信。现代个人计算机结构更加复杂，包含多重总线，我们将在后面讨论。目前，这一模式还是够用的。在下面各小节中，我们将简要地介绍这些部件，并且讨论一些操作系统设计师所考虑的硬件问题。毫无疑问，这是一个非常简要的概括介绍。

![简单个人计算机中的一些部件]({{site.url}}/assets/image/2020-09-26-OS/image1-6.png)

### 处理器

### 存储器

### 磁盘

### I/O设备

### 总线

## 	操作系统大观园

### 大型机操作系统

在操作系统的高端是用于大型机的操作系统，这些房间般大小的计算机仍然可以在一些大型公司的数据中心见到。这些计算机与个人计算机的主要差别是其I/O处理能力。一台拥有1000个磁盘和几百万吉字节数据的大型机是很正常的，如果有这样一台个人计算机朋友会很羡慕。大型机也在高端的web服务器、大型电子商务服务站点和事务一事务交易服务器上有某种程度的卷土重来。

用于大型机的操作系统主要面向多个作业的同时处理，多数这样的作业需要巨大的I/O能力。系统主要提供三类服务：批处理、事务处理和分时。批处理系统处理不需要交互式用户干预的周期性作业。保险公司的索赔处理或连锁商店的销售报告通常就是以批处理方式完成的。事务处理系统负责大量小的请求，例如，银行的支票处理或航班预订。每个业务量都很小，但是系统老须每秒处理成百上千个业务。分时系统允许多个远程用户同时在计算机上运行作业一，如在大型数据库上的查询。这些功能是密切相关的，大型机操作系统通常完成所有这些功能。大型机操作系统的一个例子是OS／390（OS/360的后继版本）。但是，大型机操作系统正在逐渐被诸如Linux这类UNIX的变体所替代。

### 服务器操作系统

下一个层次是服务器操作系统。它们在服务器上运行，服务器可以是大型的个人计算机、工作站，甚至是大型机。它们通过网络同时为若干个用户服务，并且允许用户共享硬件和软件资源。服务器可提供打印服务、文件服务或Web服务。Internet提供商运行着许多台服务器机器，为用户提供支持，使Web站点保存Web页面并处理进来的请求。典型的服务器操作系统有Solaris、FreeBSD、Linux和WindowsServer201x。
	
### 多处理器操作系统

### 个人计算机操作系统

### 掌上计算机机操作系统

### 嵌入式操作系统

### 传感器节点操作系统

### 实时操作系统

### 智能卡操作系统

## 操作系统概念

### 进程

### 地址空间

### 文件

### 输入输出

### 保护

### shell
# Onboarding prompt for a teammate's Claude Code

Paste the block below into Claude Code, started in the root of your clone of this repository.

```text
我们这个项目（ImageTrading-CV-Project）有两个人各自用 Claude Code 开发。为了让两边的 Claude 共享上下文，仓库里新增了一套共享记忆机制。请你完成接入，并把你这边的记忆合并进去。

一、先了解机制（只读）
1. 运行 git fetch origin，然后把 origin/main 合并进当前分支（或者 rebase 到它上面）。如果有未提交的改动导致无法合并，先停下来告诉我，不要 stash、reset 或丢弃任何改动。
2. 阅读 CLAUDE.md 的 "Shared memory" 一节、scripts/memory_sync.sh 和 .claude/settings.json。
3. 运行 scripts/memory_sync.sh start。它会把 memory 分支检出到 memory/ 目录并打印当前内容。
4. 阅读 memory/README.md、STATUS.md、DECISIONS.md、EXPERIMENTS.md 和 memory/log/jeffery.md。

要点：memory/ 是 memory 分支的一个 git worktree，和代码分支无关，在任何分支上内容都一样。会话开始时钩子自动拉取，每轮结束时钩子自动提交并推送。规则写在 main 的 CLAUDE.md，事实、状态和历史写在 memory/。

二、设置身份
运行 git config memory.author charles。之后你只写 memory/log/charles.md，不要改 memory/log/jeffery.md。

三、把你这边的记忆合并进来
目前 memory/ 里只有 Jeffery 那边的内容。请从以下来源收集你这边的信息：你本机的 Claude 记忆目录（~/.claude/projects/ 下对应本项目的 memory/）、本项目里已有的 CLAUDE.md 或笔记、最近的会话记录，以及仓库和实验产物本身。然后：
1. memory/log/charles.md：按日期补写工作日志，旧的可以压缩成每次几行。写清做了什么、发现了什么、哪些没有验证。
2. memory/EXPERIMENTS.md：核对标了 ? 的行，那是 Jeffery 的 Claude 只看汇总文件填的。改正错误、去掉 ?，补上缺的实验。
3. memory/STATUS.md：改写 "Modeling (charles)" 一节，回答其中 "To be confirmed" 和 "Open questions" 里与你有关的问题，特别是：你的 K 线数据是怎么生成和复权的；训练代码在哪里、何时能推送到仓库。
4. memory/DECISIONS.md：补上你这边已经定下来的决定和原因。对标了 "proposed; charles to confirm" 的条目，写明同意，或者写出不同意的理由，不要直接删除。
5. 如果你这边有 Claude 必须遵守的规则，先告诉我，由我决定是否加进 main 的 CLAUDE.md。

四、必须遵守
- 这个仓库是公开的。不要写入密码、token、账号名、NetID、邮箱、本机路径、私有链接，也不要写入任何 WRDS 授权数据。汇总统计可以写。
- memory/ 里的内容用英文写，保持简短。STATUS.md 控制在 60 行左右。
- 不要删除或改写 Jeffery 写的内容；有不同意见就在旁边注明。

五、完成后
运行 scripts/memory_sync.sh sync。如果报告冲突，在 memory/ 里解决后再运行一次。最后告诉我：你合并了哪些内容，哪些信息没找到，哪些地方和 Jeffery 那边的记录不一致。

另外提醒我：钩子在新开的会话里才会生效，而且第一次需要我在 Claude Code 里确认信任这个项目的钩子。
```

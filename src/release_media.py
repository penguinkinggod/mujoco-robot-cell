"""把 README 用的图片传到 GitHub Release，同名先删再传，一个文件永远只占一份。

为什么放 Release 而不是仓库里：
    GIF 每改一版在 git 历史里就多一个 blob，删不掉也压不干净，
    `clone` 要全拖下来。Release 附件存在另一套存储里，不进 git 对象库。

为什么必须同名替换：
    Release 附件没有版本历史，同名上传等于换掉那一个，下载 URL 也不会变。
    每版换个新名字（now_feed_v2.gif / now_feed_final.gif）才会越堆越多。

用法：
    python src/release_media.py --list
    python src/release_media.py assets/now_feed.gif
    python src/release_media.py assets/now_feed.gif --dry-run     # 只看会做什么
    python src/release_media.py --prune                          # 删掉 README 没引用的
    python src/release_media.py --prune --dry-run

令牌从 git 自己的凭据里取（`git credential fill`），不落盘、不进命令行历史、
不写进环境变量。前提是这台机器 push 过这个仓库。
"""
import argparse
import json
import mimetypes
import os
import subprocess
import sys
import urllib.error
import urllib.request

DEFAULT_TAG = "media-v1"
API = "https://api.github.com"
UPLOAD = "https://uploads.github.com"
README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "README.md")


# ---------- git / 凭据 ----------

def git(*args, **kw):
    return subprocess.run(("git",) + args, capture_output=True, text=True, **kw)


def repo_slug():
    r = git("remote", "get-url", "origin")
    if r.returncode != 0:
        sys.exit("找不到 origin 远端：%s" % r.stderr.strip())
    url = r.stdout.strip()
    # git@github.com:owner/repo.git  |  https://github.com/owner/repo.git
    tail = url.split("github.com")[-1].lstrip(":/")
    return tail[:-4] if tail.endswith(".git") else tail


def token():
    """从 git 凭据里读。读不到就报错，别让它去弹交互式输入。"""
    r = subprocess.run(["git", "credential", "fill"],
                       input="protocol=https\nhost=github.com\n\n",
                       capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1]
    sys.exit("取不到 GitHub 令牌：这台机器上没有 push 过该仓库的凭据。")


# ---------- HTTP ----------

def call(method, url, token_, data=None, ctype=None):
    headers = {"Authorization": "token " + token_,
               "Accept": "application/vnd.github+json"}
    if ctype:
        headers["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read()
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            msg = json.loads(body).get("message", body)
        except Exception:
            msg = body
        return e.code, {"message": msg}


def find_release(token_, repo, tag):
    code, d = call("GET", "%s/repos/%s/releases/tags/%s" % (API, repo, tag), token_)
    if code == 404:
        sys.exit("没有 tag 为 %s 的 release。先建一个（见 HANDOFF §25）。" % tag)
    if code != 200:
        sys.exit("查 release 失败 HTTP %d：%s" % (code, d.get("message")))
    return d


def assets_of(token_, repo, rel_id):
    code, d = call("GET", "%s/repos/%s/releases/%s/assets?per_page=100"
                   % (API, repo, rel_id), token_)
    if code != 200:
        sys.exit("列附件失败 HTTP %d：%s" % (code, d.get("message")))
    return d


def delete_asset(token_, repo, asset):
    code, d = call("DELETE", "%s/repos/%s/releases/assets/%s"
                   % (API, repo, asset["id"]), token_)
    if code != 204:
        sys.exit("删 %s 失败 HTTP %d：%s" % (asset["name"], code, d.get("message")))
    print("  删掉 %s（%.2f MB）" % (asset["name"], asset["size"] / 1048576))


def upload(token_, repo, rel_id, path, name):
    size = os.path.getsize(path)
    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        blob = f.read()
    url = "%s/repos/%s/releases/%s/assets?name=%s" % (UPLOAD, repo, rel_id, name)
    print("  上传 %s（%.2f MB，%s）" % (name, size / 1048576, ctype))
    code, d = call("POST", url, token_, data=blob, ctype=ctype)
    if code != 201:
        sys.exit("上传失败 HTTP %d：%s" % (code, d.get("message")))
    return d


# ---------- README 白名单 ----------

def readme_assets(tag):
    """README 里实际引用了哪些附件名。清理时只留这些。"""
    names = set()
    if not os.path.exists(README):
        return names
    with open(README, encoding="utf-8") as f:
        for line in f:
            marker = "/releases/download/%s/" % tag
            if marker in line:
                tail = line.split(marker, 1)[1]
                for sep in (")", " ", "?", '"', ">"):
                    tail = tail.split(sep, 1)[0]
                if tail:
                    names.add(tail)
    return names


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description="Release 附件：同名替换，不堆积")
    ap.add_argument("file", nargs="?", help="要上传的文件")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="release tag（默认 %s）" % DEFAULT_TAG)
    ap.add_argument("--name", help="附件名（默认用文件名，别加版本后缀）")
    ap.add_argument("--list", action="store_true", help="只列出附件")
    ap.add_argument("--prune", action="store_true", help="删掉 README 没引用的附件")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不动手")
    args = ap.parse_args()

    tok = token()
    repo = repo_slug()
    rel = find_release(tok, repo, args.tag)
    cur = assets_of(tok, repo, rel["id"])

    if args.list or not (args.file or args.prune):
        print("%s @ %s   附件 %d 个，共 %.2f MB"
              % (repo, args.tag, len(cur), sum(a["size"] for a in cur) / 1048576))
        for a in cur:
            print("  %-24s %8.2f MB  id=%s" % (a["name"], a["size"] / 1048576, a["id"]))
        total = sum(a["size"] for a in cur) / 1048576
        print("\n  合计 %.2f MB（更换时同名替换，这个数不会随重渲次数涨）" % total)
        return

    if args.file:
        if not os.path.exists(args.file):
            sys.exit("文件不存在：%s" % args.file)
        name = args.name or os.path.basename(args.file)
        if args.dry_run:
            print("[dry-run] 会先删同名 %s，再传 %.2f MB"
                  % (name, os.path.getsize(args.file) / 1048576))
        else:
            print("替换 %s：" % name)
            for a in cur:
                if a["name"] == name:
                    delete_asset(tok, repo, a)
            d = upload(tok, repo, rel["id"], args.file, name)
            print("  完成：%s  %.2f MB" % (d.get("browser_download_url"), d["size"] / 1048576))

    if args.prune:
        keep = readme_assets(args.tag)
        stale = [a for a in assets_of(tok, repo, rel["id"]) if a["name"] not in keep]
        print("\n清理：README 引用 %s，其余 %d 个可删"
              % ("、".join(sorted(keep)) or "（无）", len(stale)))
        for a in stale:
            if args.dry_run:
                print("  [dry-run] 会删 %s（%.2f MB）"
                      % (a["name"], a["size"] / 1048576))
            else:
                delete_asset(tok, repo, a)

    left = assets_of(tok, repo, rel["id"])
    print("\n现在 %d 个附件，共 %.2f MB"
          % (len(left), sum(a["size"] for a in left) / 1048576))


if __name__ == "__main__":
    main()

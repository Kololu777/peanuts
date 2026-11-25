# Peanuts

OpenAI **dots** に着想を得た、**親エージェント（Root Mediator）** が人間と複数の **子エージェント（Claude Code CLI / Codex CLI）** を仲介するマルチエージェント・オーケストレーターです。

- **サブスク認証のまま動作**: APIキー課金ではなく、ログイン済みの `claude` CLI (`claude -p --output-format json --resume`) と `codex` CLI (`codex exec --json` / `resume`) をサブプロセス実行します（オフライン検証用の `mock` プロバイダ同梱）。
- **再帰的なエージェント間委譲**: 親（Claude/Codex）が複数テーマを子（Claude/Codex）へ並列振り分けし、子も最大深度 `--max-depth` の範囲内でさらに孫エージェントを呼べます。
- **SQLite 永続化マルチターン**: エージェントツリー、プロバイダの `session_id` / `thread_id`、会話履歴、委譲状態、質問、イベントログ（`seq` 付き）を SQLite に保存します。
- **Web + React 即時移行構成**: FastAPI（REST + WebSocket + OpenAPI）を分離し、Textual TUI も完全に同じ API クライアント経由で動作します。
- **TLA+ による事前検証**: 状態機械と並行制御を [spec/PeanutsOrchestrator.tla](spec/PeanutsOrchestrator.tla) でモデル検査（TLC）済みです。

設計の詳細は [docs/architecture.md](docs/architecture.md)（状態遷移図・シーケンス図・TLA+ 検証結果）を参照してください。

---

## インストール（`pipx` / `uvx` / `uv tool`）

`pyproject.toml` に `[project.scripts] peanuts = "peanuts.cli:main"` を定義しているため、以下いずれでも利用できます。

```bash
# 1. pipx でインストール（ローカルリポジトリから）
pipx install .

# 2. uv を使う場合（uv tool install は pipx 相当の常駐インストール）
uv tool install .

# 3. インストールせず一時実行したい場合（uvx）
uvx --from . peanuts --provider mock
```

> **`pipx` vs `uvx` / `uv tool install` の使い分け**:
> すでに `uv` を使っているなら、常駐インストールは **`uv tool install`**（`pipx install` と同等かつ高速）、ワンショット試用は **`uvx`** がもっとも快適です。配布パッケージとしては `pyproject.toml` の標準形式なので `pipx` でも `uvx` でもそのまま動きます。

---

## クイックスタート

```bash
# 利用可能な CLI プロバイダ確認 (claude / codex / mock)
peanuts providers

# 1. TUI を起動（バックグラウンドでローカル FastAPI サーバーを自動起動）
peanuts --provider mock          # まずは Mock で複数テーマ並列委譲・再帰・質問ループを体験
peanuts --provider claude        # 親に Claude Code CLI を使用
peanuts --provider codex         # 親に OpenAI Codex CLI を使用

# 2. サーバー単体起動（Web + React 開発時や、別ターミナルの TUI から接続したい時）
peanuts serve --host 127.0.0.1 --port 8765 --provider claude
peanuts tui --server-url http://127.0.0.1:8765

# 3. React 側で型生成するための OpenAPI スキーマ出力
peanuts openapi --out openapi.json
```

### TUI でのコマンド例

- 通常入力: 選択中のエージェント（デフォルトは `root` 親エージェント）へ送信
- Mock モードでの動作デモ:
  - `auth: JWT検証を修正; docs: README更新` → 複数テーマへ並列委譲
  - `recursive: 機能実装とテスト` → 親 → 子 → 孫 の再帰委譲
  - `db: 確認 マイグレーション実行` → 子から人間への確認質問（`/answer はい` で再開）
- `/answer <回答>` / `/answer <q_id> <回答>`: `WAITING_HUMAN` の質問に回答
  - エージェントが `ask_human` に `options`（`label` / `description` / `recommended`）を付けた場合は選択肢リストを表示。入力欄が空のとき `↑`/`↓` で選び `Enter` で回答（おすすめには ★ 印。`/answer 2` の番号指定・クリック・自由記述も可）
- `/provider <claude|codex|mock>`: 選択中エージェントのプロバイダ切替
- `/retry [agent_id]`: `FAILED` エージェントを `IDLE` に復帰
- `/cancel <delegation_id>`: 委譲をキャンセル
- `/root`: 送信先を親エージェントに戻す

---

## 開発・検証コマンド

```bash
# 依存同期
uv sync --python 3.12

# 型チェック (ty)
ty check

# テスト実行 (pytest)
uv run pytest

# TLA+ モデル検査 (tla2tools.jar が手元にある場合)
java -cp /tmp/tla2tools.jar tlc2.TLC -config spec/PeanutsOrchestrator.cfg spec/PeanutsOrchestrator.tla
```

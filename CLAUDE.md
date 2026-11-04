# CLAUDE.md

このリポジトリの作業指示は AGENTS.md に集約しています（他のエージェントと共通）。

@AGENTS.md

## Claude Code 固有の補足

- コマンドは必ず `uv run ...` 経由で実行する（システムの Python には依存が入っていない）。
- `uv run wsp all` や `wsp exp2`（約 7 分）は長いので、バックグラウンド実行して完了通知を待つ。
  ポーリングのために `sleep` を挟まない。
- 実験ロジックを変更したら、コミット前に `uv run pytest -q` → `uv run wsp all` → README の結果表の更新、の順で行う。
- `--agent claude` の実行は API 課金が発生するため、ユーザーの明示的な指示がある場合のみ行う。
- 一時ファイルはリポジトリ内に作らない（`artifacts/` はキャッシュ専用）。

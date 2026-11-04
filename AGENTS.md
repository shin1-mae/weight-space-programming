# AGENTS.md — weight-space-programming

AI コーディングエージェント（Claude Code / Codex / Copilot など）向けの作業指示書です。
人間向けの概要は [README.md](README.md)、設計は [docs/architecture.md](docs/architecture.md) を参照してください。

## プロジェクト概要

LLM（または決定的な参照エージェント）が **学習ループを回さずに** ニューラルネットの重みを
Read（読解）/ Fix（外科的修正）/ Write（直接生成）する「Weight-Space Programming」のハーネスと、
製造業ドメイン（3D CAD の SDF ネットワーク、図面の穴指示分類器）での 3 つの検証実験です。

- Exp1 `wsp exp1`: SDF-MLP の重みから部品形状・穴の設計意図を読む（Read）
- Exp2 `wsp exp2`: 設計変更指示（ECN）を重み空間の厳密な演算で反映する（Write / Edit）
- Exp3 `wsp exp3`: はめあい公差を無視する分類器を重みの直接編集でホットフィックス（Fix）

## リポジトリ構成

```text
src/wsp/
  harness/      inspector.py（読む）/ editor.py（書く・ロールバック・パッチ）/ sandbox.py + _sandbox_worker.py（隔離実行）/ tools.py（型付きツール）
  geometry/     parts.py（パラメトリック部品と解析 SDF）/ mesh.py（メッシュ化と評価指標）/ sdf_ops.py（重みプログラム演算）
  models/       cad_sdf_mlp.py（SDF-MLP とフィット）/ drawing_head.py（図面穴分類器）
  data/         drawings.py（テナント A/B の合成穴指示データ）
  agents/       llm.py（Claude ツールループ）/ scripted.py（参照エージェント）/ prompts.py / snippets/read_sdf.py（サンドボックス投入コード）
  experiments/  exp1_* / exp2_* / exp3_* / common.py
tests/          pytest（ネットワーク不要・API キー不要）
results/        `uv run wsp all` の出力（コミット対象。手編集禁止）
artifacts/      フィット済みモデル等のキャッシュ（gitignore）
docs/           architecture.md, industrial_application_notes.md
```

## セットアップとコマンド

```bash
uv sync --extra llm          # 依存関係（.venv）
uv run pytest -q             # テスト（約 1 分、CPU のみ）
uv run ruff check . && uv run ruff format --check .
uv run wsp all               # 全実験（scripted エージェント、CPU で約 10 分。初回は SDF フィットを含む）
uv run wsp exp3 --seeds 0    # 個別実験
uv run wsp exp1 --agent claude   # Claude で実行（認証情報が必要。課金が発生する）
docker compose run --rm closed-net exp3   # ネットワーク無効（閉域網想定）で実行
```

- Python 3.11+、依存管理は uv。`uv.lock` は手で編集しない（`uv lock` で更新）。
- 長時間の実験はバックグラウンド実行し、完了を待ってから結果を確認する。

## 守るべき不変条件（重要）

1. **WSP の経路で勾配を使わない。** 重みの変更は `WeightEditor`（固定形状）または
   `sdf_ops` / `SDFProgram`（構造変更）経由のみ。勾配降下が許されるのは
   「顧客モデルの作成」（`fit_sdf`, `train_drawing_head`）と比較用ベースライン
   （Exp2 `retrain`, Exp3 `finetune`）だけ。
2. **エージェントは `ToolRegistry` 経由でのみモデルに触れる。** scripted エージェントも LLM と同じツールを使う
   （モデルオブジェクトへ直接アクセスしない）。新しい能力が必要ならツールを追加する。
3. **LLM が書いたコードはプロセス内で `exec` しない。** 必ず `PythonSandbox`（別プロセス・AST 検査・
   rlimit・最小環境変数）で実行する。許可モジュールの追加は理由を明記し、`tests/test_sandbox.py` を更新。
   API キー等を子プロセスの環境変数に渡さない。
4. **`sdf_ops` は厳密であること。** 新しい演算には numpy 参照実装との一致テスト（`atol≈1e-10`）を追加する。
5. **単位系**: 公開 API・ツール入出力・結果はミリメートル。ネットワーク内部は正規化座標
   （`MM_PER_UNIT = 50`、作業空間 `[-1, 1]^3`）。
6. **結果の整合性**: `results/` は生成物。実験ロジック・既定シード・設定を変えたら `uv run wsp all` で再生成し、
   README の「実験結果サマリー」を `results/*/summary.json` と一致させる。数値を手で書き換えない。
7. **正直な報告**: scripted エージェントの結果を LLM の結果として記述しない。未実行のもの（例: Claude での実験）は
   未実行と明記する。限界（例: Exp2 の |∇f|≠1 によるオフセット誤差、Exp3 の F7/JS7 未学習トークン）を消さない。
8. **決定性**: 乱数はシード固定。テストは決定的に保つ。

## Claude API を触るとき（`src/wsp/agents/llm.py`）

- 公式 Anthropic Python SDK（`anthropic`）を使う。既定モデルは `claude-opus-5-5`（`WSP_CLAUDE_MODEL` で上書き）。
- このモデルでは thinking は常に有効。深さは `output_config={"effort": ...}` で制御し、`thinking` の無効化や
  `budget_tokens`、`temperature` 等のサンプリング引数は送らない（400 になる）。
- `tool_choice` の強制（`any` / `tool`）は使わない。終了ツール（`submit_*` / `finish`）はプロンプトで指示する。
- プロンプトキャッシュのため、`system` と `tools` は実行中に変えず、`messages` は追記のみ（過去ターンを編集しない）。
- 拒否時のサーバー側フォールバック（`fallbacks="default"`, beta `server-side-fallback-2026-07-01`）は既定で有効。
- SDK の挙動は変わりやすい。仕様に自信がないときは公式ドキュメントで確認し、推測で書かない。

## テスト方針

- テストはネットワーク・API キーなしで動くこと。LLM ループは `tests/test_llm_agent.py` のフェイククライアントで検証する。
- SDF の学習を伴う重いテストは避け、解析的に *書いた* ネットワーク（`write_part_layers`）で読解・編集を検証する。
- 数秒以上かかるテストには `@pytest.mark.slow` を付ける。
- 変更後は少なくとも `uv run pytest -q` と `uv run ruff check .` を通してからコミットする。

## コードスタイル

- ruff（line-length 120）。型ヒント必須。数式・重み操作の根拠は docstring に書く（例: `b1' = b1 - W1 t`）。
- ツール入力は `ToolInput`（Pydantic, `extra="forbid"`）で定義し、`Field(description=...)` を必ず付ける
  （LLM が読む仕様そのもの）。
- 例外はツール層で `is_error` の結果に変換される。ハンドラ内で握りつぶさない。

## 拡張の手順

- **ツール追加**: `ToolInput` サブクラス → ハンドラ → `reg.add(name, description, model, handler)`。
  予算を消費する処理は `ForwardBudget.consume` を通す。テストを追加。
- **実験追加**: `src/wsp/experiments/expN_*.py` に `run()` / `main()` / `to_markdown()` を実装し、
  `src/wsp/cli.py` の `COMMANDS` に登録。結果は `results/expN/` に `summary.json`・`cases.json`・`README.md`。

## やってはいけないこと

- `artifacts/`、大きなバイナリ、API キー、`.env` をコミットしない。
- 実在の顧客データや機密図面をリポジトリに入れない（データはすべて合成）。
- 結果を良く見せるためにシードを選び直したり、評価セットをエージェントに見せたりしない
  （Exp3 のテストセットはエージェント非公開が前提）。

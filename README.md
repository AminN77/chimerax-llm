# chimerax-llm

ChimeraX bundle that turns natural language into ChimeraX commands using an LLM agent. Supports any OpenAI-compatible API (OpenRouter, OpenAI, etc.) or [opencode](https://github.com/anomalyco/opencode) for zero-cost usage via GitHub Copilot.

## Prerequisites

- **ChimeraX** 1.1 or newer (graphical interface; does not run in `--nogui` mode).
- **Git** (to clone), or download the source as a ZIP.

**For API mode (default):**
- An **API key** for an OpenAI-compatible service.
- The bundle declares a dependency on the Python **`openai`** package; ChimeraX should install it automatically. If you see import errors, run `devel pip install openai` from the ChimeraX command line and restart.

**For opencode mode (free via GitHub Copilot):**
- **opencode** CLI installed and authenticated (see [opencode setup](#opencode-setup) below).
- A **GitHub Copilot** subscription (individual, business, or enterprise).

## Install from source

1. Clone the repository:

   ```bash
   git clone https://github.com/AminN77/chimerax-llm.git
   cd chimerax-llm
   ```

2. Start **ChimeraX** and open the **Command Line**.

3. Install the bundle:

   ```text
   devel install /full/path/to/chimerax-llm
   ```

   For development (pick up Python edits after restart without reinstalling):

   ```text
   devel install /full/path/to/chimerax-llm user true editable true
   ```

4. **Restart ChimeraX**.

## Configuration

Open the **ChimeraLLM** tool, click **Settings**, and configure one of the two backends:

### Option A: OpenAI-compatible API

| Setting | Description |
|---|---|
| **API endpoint URL** | Leave empty for OpenRouter (`https://openrouter.ai/api/v1`), or enter any OpenAI-compatible endpoint |
| **API key** | Your API key for the chosen endpoint |
| **Model** | Model identifier (default: `gpt-4o`) |
| **Temperature** | Sampling temperature, 0.0 - 2.0 (default: 0.2) |

### Option B: opencode (GitHub Copilot)

Check **"Use opencode instead of API"** in Settings. This routes all LLM calls through the opencode CLI, which connects to your GitHub Copilot subscription — no API key or billing needed.

| Setting | Description |
|---|---|
| **opencode model** | Pick from the dropdown or type a model ID. Click **Refresh models** to fetch available models from opencode. |

Available GitHub Copilot models include Claude (Sonnet, Opus, Haiku), GPT-4o/5, Gemini, Grok, and more. Notably, GitHub Copilot provides free-tier access to models like GPT-4.1, GPT-4o, and GPT-5-mini at no additional cost beyond your Copilot subscription. opencode also offers its own free community models.

**Max iterations** (shared setting) controls how many tool-calling rounds the agent can take per message (default: 10).

## opencode setup

1. **Install opencode** (requires Node.js 18+):

   ```bash
   npm i -g opencode-ai
   ```

   Or via Homebrew:

   ```bash
   brew install opencode-ai/tap/opencode
   ```

2. **Authenticate with GitHub Copilot:**

   ```bash
   opencode
   ```

   On first run, opencode will prompt you to log in via GitHub OAuth. Follow the device code flow to authorize.

3. **Verify it works:**

   ```bash
   opencode models github-copilot
   ```

   You should see a list of available models. If not, re-run `opencode` and complete the login flow.

4. In ChimeraX, open **ChimeraLLM Settings**, check **"Use opencode instead of API"**, click **Refresh models**, pick a model, and save.

> **Note:** ChimeraX launches as a macOS GUI app and may not inherit your shell's PATH. The plugin searches common install locations (`/opt/homebrew/bin`, `/usr/local/bin`, `~/.local/bin`) automatically. If opencode still isn't found, you can create a symlink: `sudo ln -s $(which opencode) /usr/local/bin/opencode`

## Usage

- **Menu:** Tools > **ChimeraLLM**
- **Command line:**

  ```text
  chimerallm
  ```

  With an inline prompt:

  ```text
  chimerallm fetch 1ubq and color it by secondary structure
  ```

Type natural language requests in the chat panel. The agent will call ChimeraX commands, inspect session state, and reply with results.

## Updating

After `git pull`, run `devel install` again with the same path and options, then restart ChimeraX.

## Architecture

See [architecture.md](architecture.md) for a detailed overview of the codebase, threading model, and backend design.

## More information

- ChimeraX `devel install` options: [Command: devel](https://rbvi.ucsf.edu/chimerax/docs/user/commands/devel.html)
- Bundle development overview: [Building and distributing bundles](https://www.cgl.ucsf.edu/chimerax/docs/devel/writing_bundles.html)
- opencode documentation: [github.com/anomalyco/opencode](https://github.com/anomalyco/opencode)

# chimerax-llm

ChimeraX bundle: LLM chat that turns natural language into ChimeraX commands (OpenAI-compatible API).

## Prerequisites

- **ChimeraX** 1.1 or newer (graphical interface; this bundle does not run in `--nogui` mode).
- **Git** (to clone this repository), or download the source as a ZIP and unpack it.
- An **API key** for an OpenAI-compatible service (e.g. OpenAI, or another provider that speaks the same HTTP API).

The bundle declares a dependency on the Python **`openai`** package; ChimeraX should install it when you install the bundle. If you see import errors for `openai`, install it into ChimeraX’s environment (e.g. `devel pip install openai` from the ChimeraX command line) and restart.

## Install from source

1. Clone the repository (or unpack the ZIP) so you have a folder that contains **`bundle_info.xml`** at its top level.

   ```bash
   git clone https://github.com/AminN77/chimerax-llm.git
   cd chimerax-llm
   ```

2. Start **ChimeraX** and open the **Command Line**.

3. Install the bundle with **`devel install`**, using the **full path** to that folder (the directory that contains `bundle_info.xml`):

   ```text
   devel install /full/path/to/chimerax-llm
   ```

   For development, you can install into your user package area and pick up Python edits after restart without reinstalling metadata:

   ```text
   devel install /full/path/to/chimerax-llm user true editable true
   ```

4. **Restart ChimeraX** after installation (required for the tool and command to register reliably).

## First-time configuration

1. Open the **ChimeraLLM** tool (see below).
2. Click **Settings**.
3. Enter your **API key**, choose **model** and options as needed, then confirm so settings are saved.

Without a saved API key, the agent will report that the key is missing.

## Usage

- **Menu:** use the ChimeraX **Tools** menu and start **ChimeraLLM** (name may match the bundle’s tool label).
- **Command line:** run:

  ```text
  chimerallm
  ```

  Optional text after the command is queued as a prompt when the tool is open:

  ```text
  chimerallm fetch 1abc
  ```

Type requests in the panel; the assistant runs ChimeraX commands via the session and shows output in the chat.

## Updating

After `git pull` (or replacing the folder with a newer ZIP), run **`devel install`** again with the same path and options, then restart ChimeraX.

## More information

- ChimeraX **`devel install`** options: [Command: devel](https://rbvi.ucsf.edu/chimerax/docs/user/commands/devel.html)
- Bundle development overview: [Building and distributing bundles](https://www.cgl.ucsf.edu/chimerax/docs/devel/writing_bundles.html)

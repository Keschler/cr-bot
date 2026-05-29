# Clash Royale Sound Effects

A structured archive of Clash Royale sound effects organized by card/unit. Files are original `.wav` exports pulled directly from the game assets; no processing or re-encoding has been applied unless noted in a specific folder.

## What’s inside
- **Organization**: Each folder corresponds to a card or feature (for example, `card_common_knight`, `card_legendary_princess`, `card_tower_kingtower`). Inside, you’ll find one or more `.wav` files named after the event/variation (e.g., `attack`, `spawn`, `hit`, `vo_...`) and a numeric suffix where there are multiple takes.
- **Format**: Files are in their original `.wav` format as extracted from the client data. Metadata is preserved as-is.

## Usage
- **Browse and preview**: Open any folder and play the `.wav` files in your audio player/DAW.
- **Attribution**: If you use these for content (videos, mods, educational demos), please credit Supercell and this archive as the source of the rip.
- **Do not redistribute commercially**: See the licensing/disclaimer section below.

## Folder naming and file naming
- **Folder names**: `card_{rarity}_{name}` or `card_evolution_{name}` follow in-game taxonomy to make discovery simple.
- **File names**: Typically `{short_context}_{index}.wav` (for example, `fire_01.wav`, `hit_03.wav`). Some legacy folders may use slightly different conventions where that matched in-game naming.

## Contributing
- **Fixes/renames**: PRs to improve naming consistency, add missing sounds, or correct classifications are welcome.
- **Additions**: If you have additional clean rips, match the existing folder structure and file naming.
- **Issues**: Open an issue if you notice corrupt files, broken links, or misclassified assets.

## Disclaimer and licensing
- This repository is a fan-made archive and is **not affiliated with or endorsed by Supercell**.
- All audio assets are the property of their respective owners (Supercell). They are provided here strictly for **educational, archival, and non-commercial** purposes.
- If you are the rights holder and would like specific files or folders removed, please open an issue or contact the maintainer and they will be taken down promptly.

## Project files (.fsproj)
Any IDE/project files that were previously present have been removed to keep the repo focused on audio assets and to avoid unnecessary metadata. If you need per-folder project files for your workflow, consider generating them locally rather than committing them.

## Notes
- Large binary assets can make Git clones heavy. If you plan to fork and expand the archive substantially, consider using Git LFS for `.wav` files.

Enjoy exploring and using the sounds!

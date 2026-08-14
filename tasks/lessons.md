# Lessons

- When a user asks for column sorting, apply it to the table they are discussing, not just adjacent picker tables. Confirm all relevant tables support the behavior before saying it is done.
- When replacing a UI framework, preserve existing status/header information and visual affordances unless explicitly asked to simplify.
- Never assign instance attributes that shadow framework lifecycle methods like Textual `App.run()`; use names like `run_data` for model state.
- When porting a TUI, preserve feature parity for alternate modes like plotext chart panels, legends, color mapping, and statusline context before calling the migration complete.

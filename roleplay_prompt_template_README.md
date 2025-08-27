# Roleplay Prompt Template

This file (`roleplay_prompt_template.txt`) contains the template used to generate roleplay questions for SpeakGenie.

## How to Customize

You can edit this file to change how the AI generates roleplay questions. The template uses a special placeholder `{scenario_context}` that gets replaced with the actual scenario name.

### Template Structure

1. **Main Instruction**: Tell the AI what to generate
2. **Requirements**: List specific guidelines for question generation
3. **Format**: Specify how the output should be formatted
4. **Example**: Provide a sample to guide the AI

### Customization Tips

- **Age Range**: Modify "children aged 6-16" to target different age groups
- **Question Count**: Change "exactly 10" to generate more or fewer questions
- **Perspectives**: Adjust the role descriptions (teacher, cashier, parent) for different scenarios
- **Safety Guidelines**: Add or modify restrictions to ensure child-appropriate content
- **Language Style**: Adjust vocabulary complexity and sentence structure requirements

### Example Modifications

**For Younger Children (3-8 years):**
```
- Questions should be appropriate for children aged 3-8
- Use very simple vocabulary and single-sentence questions
- Include more visual or action-based prompts
```

**For Older Students (12-18 years):**
```
- Questions should be appropriate for teenagers aged 12-18
- Use more complex vocabulary and longer conversations
- Include real-world scenarios and problem-solving
```

### Testing Changes

After modifying the template:
1. Save the file
2. Restart the SpeakGenie application
3. Try generating questions for a roleplay scenario
4. Check if the AI follows your new guidelines

### Fallback Behavior

If this file cannot be loaded, SpeakGenie will use a built-in default template. The application will show a warning message if the file is missing or corrupted.

## Technical Notes

- File encoding: UTF-8
- Placeholder format: `{scenario_context}` (required)
- File location: Same directory as `streamlit-app.py`
- Backup: Keep a copy of your customized template before major changes

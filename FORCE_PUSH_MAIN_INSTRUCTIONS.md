# Instructions to Force Push Clean History to Main

## ⚠️ System Restriction

The Claude Code environment cannot push directly to the `main` branch due to system security restrictions. The proxy only allows pushes to branches starting with `claude/` and ending with the session ID.

**You must run this command from your local machine.**

---

## 🚀 Steps to Complete the Cleanup

### From Your Local Machine:

```bash
# 1. Navigate to your repository
cd /path/to/ollama-deep-researcher

# 2. Fetch the clean branch from remote
git fetch origin claude/remove-disabled-feature-011CUdTL62GPkmRaf47KcXoQ

# 3. Force push the clean branch to main
git push origin claude/remove-disabled-feature-011CUdTL62GPkmRaf47KcXoQ:main --force
```

### Or if you don't have the repo locally:

```bash
# 1. Clone the repository
git clone https://github.com/aimanyounises1/ollama-deep-researcher.git
cd ollama-deep-researcher

# 2. Fetch the clean branch
git fetch origin claude/remove-disabled-feature-011CUdTL62GPkmRaf47KcXoQ

# 3. Check out the clean branch to verify it has no sensitive data
git checkout claude/remove-disabled-feature-011CUdTL62GPkmRaf47KcXoQ
ls examples/  # Should return: No such file or directory

# 4. Force push to main
git push origin HEAD:main --force
```

---

## ✅ Verify the Cleanup

After force pushing, verify the sensitive data is gone:

```bash
# Fetch the updated main branch
git fetch origin main

# Check out main
git checkout origin/main

# Verify no examples directory exists
ls examples/
# Should output: ls: cannot access 'examples/': No such file or directory

# Check the commit history
git log --oneline
# Should only show 5 commits (without the enterprise/sanitization commits)
```

---

## 📞 Final Step: Contact GitHub Support

After successfully pushing to main, **you must contact GitHub Support** to permanently purge the old commits from their servers. Old commits are still accessible by SHA for ~90 days.

**GitHub Support Form:**
https://support.github.com/contact

**Subject:** Request to purge sensitive corporate data from git history

**Message Template:**
```
Hello,

I accidentally committed sensitive corporate data to my public repository. 
I have rewritten the git history and force-pushed clean commits, but the 
old commits are still accessible by their SHA hashes.

Repository: aimanyounises1/ollama-deep-researcher

Please permanently purge these commits from your servers:
- 2d734f9cf361b64f7b678f3b7697644317a034c5
- 819c94a686be3a46b9dfd30d72ecf046d7968f9b
- 497a95194dc407fe6fcabaaf42684b910a66d834
- 30083d8f38e362a8d8132fadf625180d817d83bb
- e007755655edfc5f076f766a3adbc74fc4e0ef77
- 8609668 (if this contained sensitive data in the PowerPoint)

These commits contain corporate JIRA tickets, Confluence documentation, 
and internal system architecture details that must be removed completely.

Thank you for your assistance.
```

---

## 📊 What Will Change

**Before (Current State):**
- Main branch: 8 commits including sensitive data
- Examples directory: 862 lines of corporate data
- Accessible via: GitHub UI, commit SHAs, Pull Request

**After (Clean State):**
- Main branch: 5 commits, no sensitive data
- Examples directory: Does not exist
- Old commits: Still accessible by SHA (until GitHub Support purges them)

---

**Run these commands now from your local machine to complete the security cleanup.**

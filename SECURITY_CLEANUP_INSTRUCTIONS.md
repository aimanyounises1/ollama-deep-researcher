# URGENT: Security Cleanup Instructions

## Current Status
✅ **Clean branch created**: `claude/remove-disabled-feature-011CUdTL62GPkmRaf47KcXoQ` 
   - Contains ONLY safe commits (no sensitive data)
   - Clean history: 3 commits only

❌ **Main branch STILL HAS SENSITIVE DATA**:
   - commit 2d734f9: example research report
   - commit 819c94a: enterprise integration research (JIRA, Confluence, Perforce data)
   - commit 497a951: "sanitized" data (still contains corporate info)

## CRITICAL: Main Branch Must Be Cleaned

### Option 1: Force Push Clean History to Main (RECOMMENDED)

1. **Temporarily disable branch protection** (if enabled):
   - Go to: https://github.com/aimanyounises1/ollama-deep-researcher/settings/branches
   - Disable protection on `main` branch

2. **Force push clean history** (run locally):
   ```bash
   git checkout claude/remove-disabled-feature-011CUdTL62GPkmRaf47KcXoQ
   git push origin HEAD:main --force
   ```

3. **Re-enable branch protection** after force push

4. **Contact GitHub Support** to purge orphaned commits:
   - Go to: https://support.github.com/contact
   - Subject: "Request to purge sensitive data from git history"
   - Message template:
   ```
   I accidentally committed sensitive corporate data to my repository.
   I have rewritten the history and force-pushed, but the old commits
   are still accessible by their SHA hashes:
   
   Repository: aimanyounises1/ollama-deep-researcher
   Commits to purge:
   - 2d734f9cf361b64f7b678f3b7697644317a034c5
   - 819c94a686be3a46b9dfd30d72ecf046d7968f9b
   - 497a95194dc407fe6fcabaaf42684b910a66d834
   - 30083d8f38e362a8d8132fadf625180d817d83bb (contains sensitive merge)
   - e007755655edfc5f076f766a3adbc74fc4e0ef77 (merge commit)
   
   Please permanently purge these commits from your servers.
   ```

### Option 2: Delete and Recreate Repository (MOST SECURE)

1. **Export clean history**:
   ```bash
   git checkout claude/remove-disabled-feature-011CUdTL62GPkmRaf47KcXoQ
   git bundle create clean-repo.bundle HEAD
   ```

2. **Delete the GitHub repository**:
   - Go to: https://github.com/aimanyounises1/ollama-deep-researcher/settings
   - Scroll to "Danger Zone" → "Delete this repository"

3. **Create new repository** with same name

4. **Push clean history**:
   ```bash
   git clone clean-repo.bundle ollama-deep-researcher-clean
   cd ollama-deep-researcher-clean
   git remote set-url origin git@github.com:aimanyounises1/ollama-deep-researcher.git
   git push -u origin main
   ```

## Why This Is Critical

Even though commits are removed from branches, GitHub keeps them accessible by SHA for ~90 days:
- https://github.com/aimanyounises1/ollama-deep-researcher/commit/2d734f9
- https://github.com/aimanyounises1/ollama-deep-researcher/commit/819c94a
- https://github.com/aimanyounises1/ollama-deep-researcher/commit/497a951

**Anyone with these SHAs can still view the sensitive data!**

## Sensitive Data That Was Exposed

The commits contain:
- ❌ JIRA ticket numbers and internal URLs
- ❌ Confluence documentation structure
- ❌ Perforce changelist numbers
- ❌ Corporate system names (ServiceGateway, ServiceBus, APIX, ESB)
- ❌ API identifiers and architecture details
- ❌ Developer usernames and dates
- ❌ Project codes and sprint information

## Next Steps

1. Choose Option 1 or Option 2 above
2. Execute the cleanup immediately
3. Contact GitHub Support to purge orphaned commits
4. Verify no sensitive data remains accessible

**Time is critical - act now to secure your corporate data!**

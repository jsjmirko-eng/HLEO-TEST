"""
Reddit collector using PRAW (OAuth2).
Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.

Status codes returned by search_with_status():
  ok             — posts retrieved successfully
  no_credentials — REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set
  auth_error     — credentials invalid or app not approved
  rate_limited   — Reddit API 429 / too many requests
  no_results     — authenticated but query returned 0 posts
  network_error  — connection timeout or DNS failure
"""
import logging
import os
from datetime import datetime
from typing import List, Tuple

from collectors.base import RawTestimonial

logger = logging.getLogger(__name__)

# Status constants
STATUS_OK             = "ok"
STATUS_NO_CREDENTIALS = "no_credentials"
STATUS_AUTH_ERROR     = "auth_error"
STATUS_RATE_LIMITED   = "rate_limited"
STATUS_NO_RESULTS     = "no_results"
STATUS_NETWORK_ERROR  = "network_error"

# Human-readable reasons for each status
STATUS_MESSAGES = {
    STATUS_NO_CREDENTIALS: (
        "Reddit credentials are not configured. Set REDDIT_CLIENT_ID and "
        "REDDIT_CLIENT_SECRET in the environment, then register a free Reddit "
        "script app at https://www.reddit.com/prefs/apps to get these values."
    ),
    STATUS_AUTH_ERROR: (
        "Reddit authentication failed — the credentials on file were rejected. "
        "Check that REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET are correct and "
        "that the Reddit app type is 'script'."
    ),
    STATUS_RATE_LIMITED: (
        "Reddit rate limit reached. Wait 60 seconds and try again."
    ),
    STATUS_NETWORK_ERROR: (
        "Could not reach Reddit's API. Check the server's network connection "
        "and try again."
    ),
    STATUS_NO_RESULTS: None,   # Filled dynamically with the query term
}


class RedditCollector:
    USER_AGENT = "hleo:v1.0 (clinical research pipeline)"

    def _make_client(self):
        """
        Build a read-only PRAW Reddit client from env secrets.
        Returns (client, None) on success or (None, status_code) on failure.
        """
        client_id     = os.getenv("REDDIT_CLIENT_ID", "").strip()
        client_secret = os.getenv("REDDIT_CLIENT_SECRET", "").strip()

        if not client_id or not client_secret:
            return None, STATUS_NO_CREDENTIALS

        try:
            import praw
            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=self.USER_AGENT,
            )
            # Force token fetch to surface auth errors early
            _ = reddit.auth.scopes()
            return reddit, None

        except ImportError:
            logger.error("praw not installed — run: pip install praw")
            return None, STATUS_NETWORK_ERROR

        except Exception as exc:
            msg = str(exc).lower()
            if "401" in msg or "403" in msg or "invalid_grant" in msg or "unauthorized" in msg:
                logger.warning(f"Reddit auth error: {exc}")
                return None, STATUS_AUTH_ERROR
            if "429" in msg or "ratelimit" in msg.replace(" ", ""):
                logger.warning(f"Reddit rate limit: {exc}")
                return None, STATUS_RATE_LIMITED
            logger.exception(f"Reddit client creation failed: {exc}")
            return None, STATUS_NETWORK_ERROR

    def search_with_status(
        self,
        query: str,
        limit: int = 15,
        subreddits: str = "all",
    ) -> Tuple[List[RawTestimonial], str, str]:
        """
        Search Reddit posts via PRAW OAuth.

        Returns:
            (posts, status_code, human_readable_reason)

        Example:
            posts, status, reason = collector.search_with_status("finasteride hair loss")
            if status != STATUS_OK:
                print(reason)
        """
        reddit, error_status = self._make_client()
        if reddit is None:
            reason = STATUS_MESSAGES.get(error_status, "Unknown error.")
            return [], error_status, reason

        try:
            sub   = reddit.subreddit(subreddits)
            posts = []

            for submission in sub.search(
                query,
                limit=limit,
                sort="relevance",
                time_filter="all",
            ):
                body = submission.selftext or ""
                # Skip deleted / removed / link-only posts
                if body in ("[deleted]", "[removed]", ""):
                    continue

                posts.append(RawTestimonial(
                    source="reddit",
                    url=f"https://www.reddit.com{submission.permalink}",
                    title=submission.title,
                    text=body,
                    author=str(submission.author) if submission.author else "[deleted]",
                    created_at=datetime.utcfromtimestamp(submission.created_utc),
                ))

            if not posts:
                reason = (
                    f"No Reddit posts with body text matched \"{query}\". "
                    "Try shorter or broader search terms."
                )
                return [], STATUS_NO_RESULTS, reason

            return posts, STATUS_OK, f"Retrieved {len(posts)} post(s) from Reddit."

        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "ratelimit" in msg.replace(" ", ""):
                return [], STATUS_RATE_LIMITED, STATUS_MESSAGES[STATUS_RATE_LIMITED]
            if "401" in msg or "403" in msg or "unauthorized" in msg:
                return [], STATUS_AUTH_ERROR, STATUS_MESSAGES[STATUS_AUTH_ERROR]
            if "timeout" in msg or "connection" in msg or "resolve" in msg:
                return [], STATUS_NETWORK_ERROR, STATUS_MESSAGES[STATUS_NETWORK_ERROR]
            logger.exception(f"Reddit search failed: {exc}")
            return [], STATUS_NETWORK_ERROR, f"Reddit search failed: {exc}"

    def search(self, query: str, limit: int = 10) -> List[RawTestimonial]:
        """
        Backwards-compatible wrapper used by the pipeline's collect() method.
        Returns an empty list silently on any error (errors logged only).
        """
        posts, status, reason = self.search_with_status(query, limit=limit)
        if status != STATUS_OK:
            logger.info(f"Reddit search silent-fail [{status}]: {reason}")
        return posts

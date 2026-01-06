# Google Photos Upload via rclone

This project can optionally upload the final output image to Google Photos after
generation using `rclone`.

The code path runs:

`rclone copy <image_path> <remote>:album/<album>`

Where `<remote>` is an `rclone` remote of type **Google Photos**.

## Install rclone

1. Install rclone for your OS: https://rclone.org/downloads/
2. Ensure `rclone` is on your PATH. Verify with: `rclone version`

## Recommended: create your own Google OAuth client (avoids shared-quota issues)

rclone can use its default/shared OAuth client, but it’s easy to run into upload
quota/rate-limit issues when many people use the same client. For reliable
automation, create your own Google Cloud project + OAuth client ID/secret and
use those when creating the `rclone` remote.

## Step-by-step: create a Google Photos remote backed by your own project

### Step 1 — Create a Google Cloud project and enable the Photos API

1. Go to Google API Console and select or create a project.
2. Enable the API:
   - APIs & Services → Library
   - Search “Photos”
   - Enable **Google Photos API / Photos Library API**

Note: You can ignore the Picker API for rclone; rclone uses the Photos Library
API surface.

### Step 2 — Configure OAuth consent screen (minimal personal setup)

1. APIs & Services → OAuth consent screen → Configure consent screen.
2. Set:
   - App name (e.g. `rclone-gphotos`)
   - Support email
   - Audience: typically **External**
   - Add yourself as a **Test user** (your Gmail address)

rclone’s “making your own client_id” flow is written for Drive, but the consent
screen mechanics are the same:
https://rclone.org/drive/#making-your-own-client-id

### Step 3 — Add the correct Google Photos scopes (important in 2025+)

In the consent-screen UI (usually Data Access → “Add or remove scopes”), add the
scopes rclone requires for a custom client:

- `https://www.googleapis.com/auth/photoslibrary.appendonly`
- `https://www.googleapis.com/auth/photoslibrary.readonly.appcreateddata`
- `https://www.googleapis.com/auth/photoslibrary.edit.appcreateddata`

These `*.appcreateddata` scopes are the key: Google removed broad library scopes,
and rclone can only manage media/albums created by the app identity tied to your
OAuth client.

### Step 4 — Create the OAuth client ID (Desktop app)

1. APIs & Services → Credentials
2. Create Credentials → OAuth client ID
3. Application type: **Desktop app**
4. Create → copy the **Client ID** and **Client secret**

Redirect URI note: if you accidentally create a **Web application** client, you
must set the redirect URI to match rclone’s local callback or you’ll get
`redirect_uri_mismatch`. rclone’s auth flow uses a local server at:

`http://127.0.0.1:53682/`

Using a Desktop app client usually avoids redirect URI fiddling.

### Step 5 — Create a new rclone remote using your client ID/secret

Run:

`rclone config`

Then:

- `n` = New remote
- Name it (example: `gphotos_personal`)
- Storage = **Google Photos**
- Google Application Client Id → paste yours
- Google Application Client Secret → paste yours
- Use web browser auth = `y`

During auth, rclone starts a local listener on `http://127.0.0.1:53682/`. If you
have a host firewall that blocks loopback listeners, temporarily allow it.

### Step 6 — Update this project’s destination

In `config/config.yaml`, set the remote name you created above:

```yaml
rclone:
  enabled: true
  remote: gphotos_personal
  album: The Day's Art
```

Verify the remote can see albums:

`rclone lsd gphotos_personal:album`

Album note: with the required `appcreateddata` scopes, rclone can only upload to
albums created by this app identity. If you switch to a new OAuth client, you
may need to let rclone create a new album under that identity (it will create
the album on first upload if it doesn’t exist).

### Step 7 — Avoid the “refresh token expires every 7 days” trap

If your OAuth consent screen stays in **Testing**, refresh tokens/grants often
expire quickly (commonly ~7 days), forcing frequent re-auth.

For personal automation, use the consent screen “Publish” / “In production”
action so tokens last longer. Google may still show “unverified app” warnings
unless you go through verification; for personal use that’s usually acceptable.

## References

- rclone Google Photos backend: https://rclone.org/googlephotos/
- rclone “make your own client_id”: https://rclone.org/googlephotos/#making-your-own-client-id

# FraudGuard AI — Fraud Intelligence Platform

A full-stack fraud detection web application built with **Streamlit**, **Supabase**, and **scikit-learn**. It provides real-time fraud analysis, 35+ ML model training, synthetic data generation, ensemble comparison, and enterprise-grade authentication with TOTP-based two-factor authentication.

---

## Live Demo

> Deploy your own instance using the guide in the [Deployment](#deployment) section.

---

## Features

### Authentication & Security
- **Supabase Auth** — email/password sign-in and sign-up
- **TOTP Two-Factor Authentication** — QR code enrollment compatible with Google Authenticator and Authy
- **Role-Based Access Control** — Admin, Researcher, and End User roles
- **Password Reset** — secure email-based reset link via Supabase
- **Local fallback mode** — works without Supabase using `users.json` + `pyotp`

### Admin Panel
- View all registered users
- Change user roles inline
- Delete users with two-click confirmation (cannot delete own account)
- Powered by Supabase Admin API with Row Level Security

### Machine Learning (35+ Models)
| Category | Models |
|---|---|
| Ensemble | Random Forest, Extra Trees, Gradient Boosting, Histogram GBM, AdaBoost, Bagging, Voting, Stacking, XGBoost, LightGBM, CatBoost |
| Linear | Logistic Regression, Ridge, SGD, Passive Aggressive, Perceptron, LDA, QDA |
| Tree | Decision Tree, Extra Tree |
| Naive Bayes | Gaussian, Bernoulli, Complement |
| Neighbors | KNN, Nearest Centroid |
| SVM | RBF, Linear, Poly, Nu-SVC, Linear SVC |
| Neural Network | MLP |
| Anomaly Detection | Isolation Forest, One-Class SVM, LOF, Elliptic Envelope |

### Other Features
- **Synthetic Fraud Data Generator** — augments real datasets using Gaussian perturbation
- **Ensemble Comparison** — trains Bagging, Boosting, and Stacking simultaneously on the same split
- **Visual Analytics** — interactive Plotly charts, confusion matrices, performance metrics
- **Reports & Export** — download predictions, ensemble results, and activity logs as CSV
- **Session Activity Log** — tracks all events during a session

---

## Screenshots

| Login Page | 2FA QR Enrollment | Dashboard |
|---|---|---|
| Email/password with Supabase Auth | Scan QR with Google Authenticator | Metrics, charts, activity feed |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | Python 3.10+ |
| Auth & MFA | Supabase Auth (TOTP MFA) |
| Database | Supabase (PostgreSQL) |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost |
| Charts | Plotly |
| TOTP (local) | pyotp |
| QR Code | qrcode + Pillow |

---

## Project Structure

```
.
├── updated.py           # Main Streamlit application
├── requirements.txt     # Python dependencies
├── users.json           # Local user store (auto-created, gitignored)
├── .gitignore
└── README.md
```

---

## Local Development

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Supabase (optional)

Create `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-public-key"
SUPABASE_SERVICE_KEY = "your-service-role-key"
```

> If you skip this step the app runs in **local mode** — authentication uses `users.json` and TOTP uses `pyotp` directly.

### 4. Run the app

```bash
streamlit run updated.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Demo accounts (local mode only)

| Username | Password | Role |
|---|---|---|
| admin | admin123 | Admin |
| researcher | research123 | Researcher |
| user1 | user123 | End User |

---

## Deployment

### Supabase Setup

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **Authentication → Sign In Methods → MFA** and enable **TOTP**
3. Go to **Project Settings → API** and copy your URL and keys
4. Run the following SQL in the **SQL Editor**:

```sql
-- Users profile table
CREATE TABLE IF NOT EXISTS public.users (
  id          uuid REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
  username    text UNIQUE NOT NULL,
  full_name   text,
  email       text,
  phone       text,
  role        text DEFAULT 'End User'
              CHECK (role IN ('Admin', 'Researcher', 'End User')),
  totp_secret text DEFAULT '',
  password    text DEFAULT '',
  created_at  timestamptz DEFAULT now()
);

-- Row Level Security
ALTER TABLE public.users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "select_authenticated"
  ON public.users FOR SELECT TO authenticated USING (true);

CREATE POLICY "update_own"
  ON public.users FOR UPDATE TO authenticated USING (auth.uid() = id);

CREATE POLICY "insert_own"
  ON public.users FOR INSERT TO authenticated WITH CHECK (auth.uid() = id);

CREATE POLICY "delete_admin_only"
  ON public.users FOR DELETE TO authenticated
  USING ((auth.jwt() -> 'user_metadata' ->> 'role') = 'Admin');
```

### Streamlit Community Cloud

1. Push your code to GitHub (make sure `requirements.txt` is committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, file `updated.py`
4. Click **Advanced settings → Secrets** and paste:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-key"
SUPABASE_SERVICE_KEY = "your-service-role-key"
```

5. Click **Deploy**
6. Copy your app URL and add it to **Supabase → Authentication → URL Configuration → Site URL**

---

## Authentication Flow

```
User visits app
      │
      ▼
  Gate 1: Login / Sign Up
      │
      ├── Supabase mode → supabase.auth.sign_in_with_password()
      └── Local mode    → hash_password() lookup in users.json
                  │
                  ▼
          Gate 2: 2FA Verification
                  │
                  ├── Supabase mode
                  │     ├── No factor enrolled → QR enrollment page
                  │     │       └── supabase.auth.mfa.enroll()
                  │     └── Factor enrolled   → TOTP verify page
                  │               └── supabase.auth.mfa.challenge() + .verify()
                  │
                  └── Local mode
                        ├── No totp_secret   → QR enrollment page (pyotp)
                        └── Secret exists    → TOTP verify page (pyotp)
                                    │
                                    ▼
                              Main Dashboard
                           (role-based navigation)
```

---

## Role Permissions

| Feature | End User | Researcher | Admin |
|---|:---:|:---:|:---:|
| Dashboard | ✅ | ✅ | ✅ |
| Synthetic Data | ✅ | ✅ | ✅ |
| Train & Predict | ✅ | ✅ | ✅ |
| Ensemble Compare | ✅ | ✅ | ✅ |
| Reports | ✅ | ✅ | ✅ |
| User Management | ❌ | ❌ | ✅ |
| Delete Users | ❌ | ❌ | ✅ |
| Change Roles | ❌ | ❌ | ✅ |

---

## Environment Modes

| Secrets configured | Auth | 2FA |
|---|---|---|
| `SUPABASE_URL` + `SUPABASE_KEY` + `SUPABASE_SERVICE_KEY` | Supabase Auth | Supabase MFA (real TOTP) |
| `SUPABASE_URL` + `SUPABASE_KEY` only | Supabase Auth | Supabase MFA (admin ops limited) |
| None | Local `users.json` | Local TOTP via `pyotp` |

---

## requirements.txt

```
streamlit
pandas
numpy
plotly
scikit-learn
xgboost
lightgbm
supabase
pyotp
qrcode[pil]
requests
```

---

## Security Notes

- Passwords are SHA-256 hashed with a salt in local mode
- TOTP secrets are stored server-side; never exposed in the UI
- Supabase Row Level Security ensures users cannot access or delete others' data without Admin role
- The `SUPABASE_SERVICE_KEY` is only used server-side (Streamlit runs on the server) and is never sent to the browser
- Admin cannot delete their own account (enforced client-side and via RLS)

---

## License

MIT License — free to use, modify, and distribute.

---

## Author

Built as a Capstone project for CIHE.

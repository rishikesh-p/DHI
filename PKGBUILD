# Maintainer: Rishikesh <your_email@example.com>
pkgname=dhi
pkgver=0.1.0
pkgrel=1
pkgdesc="A hybrid local/cloud AI OS agent in a Bubblewrap sandbox"
arch=('any')
url="https://github.com/rishikesh/dhi"
license=('MIT')
depends=('python' 'bubblewrap' 'ollama' 'alsa-lib')
makedepends=('python-pip' 'git')

# For the AUR, you would change this source to point to your GitHub release tarball
# e.g., source=("$pkgname-$pkgver.tar.gz::https://github.com/rishikesh/dhi/archive/refs/tags/v$pkgver.tar.gz")
source=("$pkgname::git+file://${PWD}")
sha256sums=('SKIP')

package() {
    cd "$srcdir/$pkgname"
    
    msg2 "Creating virtual environment in /opt/$pkgname..."
    install -d "$pkgdir/opt/$pkgname"
    python -m venv "$pkgdir/opt/$pkgname/venv"
    
    msg2 "Installing dependencies into the venv..."
    # We install directly into the sandbox venv so we don't pollute the user's system Python
    # and we don't have to rely on the AUR having every single langchain dependency available.
    "$pkgdir/opt/$pkgname/venv/bin/pip" install --no-cache-dir .
    
    msg2 "Creating executable wrapper in /usr/bin..."
    install -d "$pkgdir/usr/bin"
    cat <<EOF > "$pkgdir/usr/bin/dhi"
#!/bin/bash
exec /opt/$pkgname/venv/bin/dhi "\$@"
EOF
    chmod +x "$pkgdir/usr/bin/dhi"
}

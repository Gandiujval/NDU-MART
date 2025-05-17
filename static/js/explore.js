document.addEventListener('DOMContentLoaded', function() {
    // Price range slider functionality
    const minPriceInput = document.getElementById('min-price');
    const maxPriceInput = document.getElementById('max-price');
    
    if (minPriceInput && maxPriceInput) {
        minPriceInput.addEventListener('change', updateFilters);
        maxPriceInput.addEventListener('change', updateFilters);
    }
    
    // Category filter functionality
    const categoryLinks = document.querySelectorAll('.dropdown-item[href*="category="]');
    categoryLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const url = new URL(this.href);
            updateFilters(url);
        });
    });
    
    // Sort functionality
    const sortLinks = document.querySelectorAll('.dropdown-item[href*="sort="]');
    sortLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const url = new URL(this.href);
            updateFilters(url);
        });
    });
    
    // Search functionality
    const searchForm = document.querySelector('form[action*="explore"]');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const searchInput = this.querySelector('input[name="search"]');
            const url = new URL(window.location.href);
            url.searchParams.set('search', searchInput.value);
            updateFilters(url);
        });
    }
    
    // Function to update filters and reload page
    function updateFilters(url = new URL(window.location.href)) {
        // Preserve existing parameters
        const currentParams = new URLSearchParams(window.location.search);
        for (const [key, value] of currentParams.entries()) {
            if (!url.searchParams.has(key)) {
                url.searchParams.set(key, value);
            }
        }
        
        // Update price range if inputs exist
        if (minPriceInput && maxPriceInput) {
            url.searchParams.set('min_price', minPriceInput.value);
            url.searchParams.set('max_price', maxPriceInput.value);
        }
        
        // Navigate to the new URL
        window.location.href = url.toString();
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Product card hover effects
    const productCards = document.querySelectorAll('.product-card');
    productCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.classList.add('shadow-lg');
        });
        
        card.addEventListener('mouseleave', function() {
            this.classList.remove('shadow-lg');
        });
    });
}); 
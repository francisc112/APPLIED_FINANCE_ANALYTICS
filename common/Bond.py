import numpy as np

class Bond:
    def __init__(self, face, coupon_rate, ytm, years, frequency=2, spread=0):
        """
        Initialize a Bond.
        :param face: Face value of the bond.
        :param coupon_rate: Annual coupon rate (in decimal, e.g., 0.05 for 5%).
        :param ytm: The base yield-to-maturity (risk-free rate component).
        :param years: Time to maturity in years.
        :param frequency: Number of coupon payments per year.
        :param spread: Credit spread (in decimal) added to the base yield.
        """
        self.face = face
        self.coupon_rate = coupon_rate
        self.ytm = ytm
        self.years = years
        self.frequency = frequency
        self.spread = spread  # Additional yield component for credit risk

    def price(self, yield_rate=None):
        """
        Calculate the bond price by discounting future cash flows.
        If yield_rate is not provided, it uses (ytm + spread).
        """
        if yield_rate is None:
            yield_rate = self.ytm + self.spread
        periods = int(self.years * self.frequency)
        coupon = self.face * self.coupon_rate / self.frequency
        price = 0
        for t in range(1, periods + 1):
            price += coupon / (1 + yield_rate / self.frequency)**t
        price += self.face / (1 + yield_rate / self.frequency)**periods
        return price

    def modified_duration(self, dy=0.0001):
        """
        Calculate the modified duration using the central finite difference method.
        """
        base_yield = self.ytm + self.spread
        P0 = self.price(base_yield)
        P_up = self.price(base_yield + dy)
        P_down = self.price(base_yield - dy)
        md = (P_down - P_up) / (2 * P0 * dy)
        return md

    def convexity(self, dy=0.0001):
        """
        Calculate convexity using the finite difference method.
        """
        base_yield = self.ytm + self.spread
        P0 = self.price(base_yield)
        P_up = self.price(base_yield + dy)
        P_down = self.price(base_yield - dy)
        conv = (P_up + P_down - 2 * P0) / (P0 * (dy ** 2))
        return conv

    def effective_duration(self, dy=0.0001):
        """
        For a non-callable bond, effective duration can be approximated by the modified duration.
        """
        return self.modified_duration(dy)
    
    def effective_convexity(self, dy=0.0001):
        """
        For a non-callable bond, effective convexity is similar to the convexity calculated above.
        """
        return self.convexity(dy)
    
    def spread_duration(self, dy=0.0001):
        """
        Calculate the spread duration by measuring the sensitivity of the bond price
        to a change in the credit spread.
        """
        base_yield = self.ytm + self.spread
        P0 = self.price(base_yield)
        P_up = self.price(base_yield + dy)
        P_down = self.price(base_yield - dy)
        spd = (P_down - P_up) / (2 * P0 * dy)
        return spd

    def duration_times_spread(self, dy=0.0001):
        """
        Calculate the product of modified duration and the credit spread.
        """
        return self.modified_duration(dy) * self.spread
    


class Bond_Portfolio:
    def __init__(self, bonds, weights):
        """
        Initialize a bond portfolio.
        :param bonds: List of Bond objects.
        :param weights: List of weights for each bond (should sum to 1).
        """
        self.bonds = bonds
        self.weights = weights

    def portfolio_price(self):
        """
        Compute the portfolio price as the weighted sum of individual bond prices.
        """
        return sum(w * bond.price() for bond, w in zip(self.bonds, self.weights))

    def portfolio_modified_duration(self, dy=0.0001):
        """
        Compute the portfolio modified duration as a market-value weighted average.
        """
        total_value = sum(bond.price() for bond in self.bonds)
        weighted_duration = sum(bond.price() * bond.modified_duration(dy) for bond in self.bonds)
        return weighted_duration / total_value

    def portfolio_convexity(self, dy=0.0001):
        """
        Compute the portfolio convexity as a market-value weighted average.
        """
        total_value = sum(bond.price() for bond in self.bonds)
        weighted_convexity = sum(bond.price() * bond.convexity(dy) for bond in self.bonds)
        return weighted_convexity / total_value

    def portfolio_effective_duration(self, dy=0.0001):
        """
        Compute the portfolio effective duration as a market-value weighted average.
        """
        total_value = sum(bond.price() for bond in self.bonds)
        weighted_eff_duration = sum(bond.price() * bond.effective_duration(dy) for bond in self.bonds)
        return weighted_eff_duration / total_value

    def portfolio_effective_convexity(self, dy=0.0001):
        """
        Compute the portfolio effective convexity as a market-value weighted average.
        """
        total_value = sum(bond.price() for bond in self.bonds)
        weighted_eff_convexity = sum(bond.price() * bond.effective_convexity(dy) for bond in self.bonds)
        return weighted_eff_convexity / total_value

    def portfolio_spread_duration(self, dy=0.0001):
        """
        Compute the portfolio spread duration as a market-value weighted average.
        """
        total_value = sum(bond.price() for bond in self.bonds)
        weighted_spread_duration = sum(bond.price() * bond.spread_duration(dy) for bond in self.bonds)
        return weighted_spread_duration / total_value

    def portfolio_duration_times_spread(self, dy=0.0001):
        """
        Compute the portfolio's Duration times Spread as a market-value weighted average.
        """
        total_value = sum(bond.price() for bond in self.bonds)
        weighted_dts = sum(bond.price() * bond.duration_times_spread(dy) for bond in self.bonds)
        return weighted_dts / total_value